import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, List
import random
import traceback

from ..utils import search_serper
from ..data_loader import load_questions, get_question_by_id, compare_answers
from ..config import SearchGymConfig, get_default_config

from .prompts import evaluate_answer, evaluate_answer_async


class SearchEnv(gym.Env):
    """
    Custom Gymnasium environment for search-based question answering.
    
    The agent can perform web searches and submit answers to questions,
    receiving rewards based on answer correctness.
    """
    
    def __init__(self, config: SearchGymConfig = None):
        """
        Initialize the Search Environment.
        
        Args:
            config: SearchGymConfig instance with all configuration settings
        """
        super().__init__()
        
        # Use provided config or default
        self.config = config or get_default_config()
        self.config.validate()
        
        # Load questions based on data configuration
        self._load_questions()
        
        # Environment state
        self.current_question = None
        self.current_question_index = 0
        self.step_count = 0
        self.episode_complete = False
        self.search_history = []  # Track search queries and results
        self.action_history = []
        self.answered_correctly = False
        self.submitted_answer = None
        self.total_reward = 0.0
        
        # Set random seed if provided
        if self.config.seed is not None:
            self.seed(self.config.seed)
        
        # Action space: discrete actions representing different action types
        # 0: [search], 1: [answer], 2: [finish]
        self.action_space = spaces.Discrete(3)
        
        # Observation space: dictionary containing session state and feedback
        self.observation_space = spaces.Dict({
            "question": spaces.Text(max_length=1024),
            "feedback": spaces.Text(max_length=2048),
            "step_count": spaces.Box(low=0, high=self.config.max_steps, shape=(), dtype=np.int32),
            "episode_complete": spaces.Box(low=0, high=1, shape=(), dtype=np.bool_),
            "answered_correctly": spaces.Box(low=0, high=1, shape=(), dtype=np.bool_)
        })
    
    def _load_questions(self):
        """Load questions based on configuration."""
        all_questions = load_questions()
        
        if self.config.data_mode == "random":
            self.questions = all_questions
        elif self.config.data_mode == "single":
            # Find the specific question
            question = get_question_by_id(self.config.data_source)
            self.questions = [question]
        elif self.config.data_mode == "list":
            # Load multiple specific questions
            self.questions = []
            for question_id in self.config.data_source:
                try:
                    question = get_question_by_id(question_id)
                    self.questions.append(question)
                except ValueError:
                    if self.config.verbose:
                        print(f"Warning: Question '{question_id}' not found, skipping")
            
            if not self.questions:
                raise ValueError("No valid questions found in data_source list")
        
        if self.config.verbose:
            print(f"Loaded {len(self.questions)} questions in {self.config.data_mode} mode")
    
    def _get_next_question(self):
        """Get the next question based on data mode."""
        if self.config.data_mode == "random":
            return random.choice(self.questions)
        else:
            # For single and list modes, cycle through questions
            question = self.questions[self.current_question_index % len(self.questions)]
            self.current_question_index += 1
            return question
    
    def reset(self, seed=None, options=None):
        """Reset the environment to start a new episode."""
        if seed is not None:
            self.seed(seed)
        
        # Reset episode state
        self.current_question = self._get_next_question()
        self.step_count = 0
        self.episode_complete = False
        self.search_history = []
        self.action_history = []
        self.answered_correctly = False
        self.submitted_answer = None
        self.total_reward = 0.0
        self.max_search_steps = self.config.max_search_steps
        self.current_search_steps = 0
        
        # Create initial observation
        observation = {
            "question": self.current_question["question"],
            "feedback": "New question loaded. You can [search] for information or [answer] directly.",
            "step_count": self.step_count,
            "episode_complete": self.episode_complete,
            "answered_correctly": self.answered_correctly
        }
        
        # Create info dictionary
        info = {
            "question_id": self.current_question["id"],
            "correct_answer": self.current_question["answer"],
            "search_history": self.search_history.copy(),
            "action_history": self.action_history.copy()
        }
        
        if self.config.verbose:
            print(f"🔄 New episode started!")
            print(f"Question: {self.current_question['question']}")
        
        return observation, info
    
    def step(self, action_input: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        if self.episode_complete:
            raise ValueError("Episode is complete. Call reset() to start a new episode.")
        
        self.step_count += 1
        action_str = str(action_input).strip()
        
        # Add action to history
        self.action_history.append(action_str)
        
        model_config = {
            "api_key": self.config.api_key,
            "model_name": self.config.model_name,
            "base_url": self.config.base_url,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout,
        }

        feedback = ""
        reward = 0.0
        
        try:
            # Process different action types
            if action_str.startswith("[search]"):
                self.current_search_steps += 1
                if self.current_search_steps > self.max_search_steps:
                    feedback = "You have reached the maximum number of search steps. Please provide an answer in your next action. Do not search again."
                    reward = 0.0
                else:
                    feedback, reward = self._handle_search_action(action_str)
            elif action_str.startswith("[answer]"):
                feedback, reward = self._handle_answer_action(action_str, model_config)
            elif action_str.startswith("[finish]"):
                feedback, reward = self._handle_finish_action(action_str)
            else:
                feedback = "Invalid action format. Please choose provide a search or answer in correct format."
                reward = 0.0
        
        except Exception as e:
            feedback = f"Error processing action: {str(e)}"
            reward = 0.0
            if self.config.verbose:
                print(f"Error in step: {e}")
                traceback.print_exc()
        
        # Apply step penalty
        step_penalty = self.config.step_penalty * self.step_count
        reward -= step_penalty
        
        # Normalize rewards if enabled
        if self.config.normalize_rewards:
            reward = max(0.0, min(1.0, reward))
        
        self.total_reward += reward
        
        # Check termination conditions
        terminated = False
        if action_str.startswith("[answer]") and self.answered_correctly:
            terminated = True
        elif action_str.startswith("[finish]"):
            terminated = True
        
        truncated = self.step_count >= self.config.max_steps
        
        if terminated or truncated:
            self.episode_complete = True
        
        # Create observation
        observation = {
            "question": self.current_question["question"],
            "feedback": feedback,
            "step_count": self.step_count,
            "episode_complete": self.episode_complete,
            "answered_correctly": self.answered_correctly
        }
        
        # Create info dictionary
        info = {
            "raw_action": action_str,
            "question_id": self.current_question["id"],
            "correct_answer": self.current_question["answer"],
            "submitted_answer": self.submitted_answer,
            "search_history": self.search_history.copy(),
            "action_history": self.action_history.copy(),
            "total_reward": self.total_reward
        }
        
        if terminated and self.config.verbose:
            if self.answered_correctly:
                print(f"🎉 Correct answer! Episode completed successfully!")
            else:
                print(f"❌ Episode ended without correct answer.")
        elif truncated and self.config.verbose:
            print(f"⏰ Episode ended due to max steps.")
        
        return observation, reward, terminated, truncated, info
    
    def _handle_search_action(self, action_str: str) -> Tuple[str, float]:
        """Handle search action and return feedback and reward."""
        query = action_str[8:].strip()  # Remove "[search]" prefix
        
        if not query:
            return "Search query is empty. Please provide a search query.", 0.0
        
        try:
            # Perform search using serper API
            search_results = search_serper(query, num=self.config.max_search_results)
            
            # Store search in history
            self.search_history.append({
                "query": query,
                "results": search_results,
                "step": self.step_count
            })
            
            feedback = f"Search results for '{query}':\n{search_results}"
            
            if self.config.verbose:
                print(f"🔍 Searched for: {query}")
            
            return feedback, 0.0  # No reward for search actions
            
        except Exception as e:
            return f"Search failed: {str(e)}", 0.0
    
    def _handle_answer_action(self, action_str: str, model_config: Dict[str, Any]) -> Tuple[str, float]:
        """Handle answer action and return feedback and reward."""
        answer = action_str[8:].strip()  # Remove "[answer]" prefix
        
        if not answer:
            return "Answer is empty. Please provide an answer.", 0.0
        
        self.submitted_answer = answer
        correct_answer = self.current_question["answer"]
        
        # Compare answers
        self.answered_correctly = compare_answers(answer, correct_answer)
        if self.config.eval_method == "llm" and not self.answered_correctly:
            feedback, judgment, reasoning = evaluate_answer(correct_answer, answer, self.current_question["question"], model_config)
            self.answered_correctly = (judgment == "Yes")
            # print(f"💭 Feedback: {feedback}")
            # print(f"💭 Judgment: {judgment}")
            # print(f"💭 Reasoning: {reasoning}")
        
        if self.answered_correctly:
            feedback = f"Correct! Your answer '{answer}' is correct."
            reward = self.config.correct_answer_reward
        else:
            feedback = f"Incorrect. Your answer '{answer}' is not correct."
            reward = self.config.incorrect_answer_reward
        
        if self.config.verbose:
            print(f"💭 Submitted answer: {answer}")
            print(f"✅ Correct!" if self.answered_correctly else f"❌ Incorrect")
        
        return feedback, reward
    
    def _handle_finish_action(self, action_str: str) -> Tuple[str, float]:
        """Handle finish action and return feedback and reward."""
        if self.answered_correctly:
            feedback = "Episode finished. You answered correctly!"
            reward = 0.0  # No additional reward for finishing after correct answer
        else:
            feedback = f"Episode finished without correct answer. The correct answer was: {self.current_question['answer']}"
            reward = 0.0
        
        if self.config.verbose:
            print("🏁 Episode finished by agent")
        
        return feedback, reward
    
    def render(self, mode="human"):
        """Render the current state of the environment."""
        if not self.current_question:
            print("No question loaded. Call reset() first.")
            return
        
        print("\n" + "="*60)
        print(f"SEARCH GYM SESSION")
        print("="*60)
        print(f"Question: {self.current_question['question']}")
        print(f"Category: {self.current_question.get('category', 'Unknown')}")
        print(f"Difficulty: {self.current_question.get('difficulty', 'Unknown')}")
        print(f"Steps taken: {self.step_count}/{self.config.max_steps}")
        print(f"Answered correctly: {self.answered_correctly}")
        
        # Show search history
        if self.search_history:
            print(f"\n🔍 Search History:")
            for i, search in enumerate(self.search_history, 1):
                print(f"  Search {i}: {search['query']}")
        
        if self.submitted_answer:
            print(f"\n💭 Submitted Answer: {self.submitted_answer}")
        
        if self.action_history:
            print(f"\nLast action: {self.action_history[-1]}")
        
        print("="*60)
    
    def close(self):
        """Clean up resources."""
        pass
    
    def seed(self, seed=None):
        """Set random seed for reproducibility."""
        np.random.seed(seed)
        random.seed(seed)
        return [seed]
    
    async def step_async(self, action_input: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment (async version)."""
        if self.episode_complete:
            raise ValueError("Episode is complete. Call reset() to start a new episode.")
        
        self.step_count += 1
        action_str = str(action_input).strip()
        
        # Add action to history
        self.action_history.append(action_str)
        
        model_config = {
            "api_key": self.config.api_key,
            "model_name": self.config.model_name,
            "base_url": self.config.base_url,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout,
        }

        feedback = ""
        reward = 0.0
        
        try:
            # Process different action types
            if action_str.startswith("[search]"):
                self.current_search_steps += 1
                if self.current_search_steps > self.max_search_steps:
                    feedback = "You have reached the maximum number of search steps. Please provide an answer in your next action. Do not search again."
                    reward = 0.0
                else:
                    feedback, reward = self._handle_search_action(action_str)
            elif action_str.startswith("[answer]"):
                feedback, reward = await self._handle_answer_action_async(action_str, model_config)
            elif action_str.startswith("[finish]"):
                feedback, reward = self._handle_finish_action(action_str)
            else:
                feedback = "Invalid action format. Please choose provide a search or answer in correct format."
                reward = 0.0
        
        except Exception as e:
            feedback = f"Error processing action: {str(e)}"
            reward = 0.0
            if self.config.verbose:
                print(f"Error in step: {e}")
                traceback.print_exc()
        
        # Apply step penalty
        step_penalty = self.config.step_penalty * self.step_count
        reward -= step_penalty
        
        # Normalize rewards if enabled
        if self.config.normalize_rewards:
            reward = max(0.0, min(1.0, reward))
        
        self.total_reward += reward
        
        # Check termination conditions
        terminated = False
        if action_str.startswith("[answer]") and self.answered_correctly:
            terminated = True
        elif action_str.startswith("[finish]"):
            terminated = True
        
        truncated = self.step_count >= self.config.max_steps
        
        if terminated or truncated:
            self.episode_complete = True
        
        # Create observation
        observation = {
            "question": self.current_question["question"],
            "feedback": feedback,
            "step_count": self.step_count,
            "episode_complete": self.episode_complete,
            "answered_correctly": self.answered_correctly
        }
        
        # Create info dictionary
        info = {
            "raw_action": action_str,
            "question_id": self.current_question["id"],
            "correct_answer": self.current_question["answer"],
            "submitted_answer": self.submitted_answer,
            "search_history": self.search_history.copy(),
            "action_history": self.action_history.copy(),
            "total_reward": self.total_reward
        }
        
        if terminated and self.config.verbose:
            if self.answered_correctly:
                print(f"🎉 Correct answer! Episode completed successfully!")
            else:
                print(f"❌ Episode ended without correct answer.")
        elif truncated and self.config.verbose:
            print(f"⏰ Episode ended due to max steps.")
        
        return observation, reward, terminated, truncated, info
    
    async def _handle_answer_action_async(self, action_str: str, model_config: Dict[str, Any]) -> Tuple[str, float]:
        """Handle answer action and return feedback and reward (async version)."""
        answer = action_str[8:].strip()  # Remove "[answer]" prefix
        
        if not answer:
            return "Answer is empty. Please provide an answer.", 0.0
        
        self.submitted_answer = answer
        correct_answer = self.current_question["answer"]
        
        # Compare answers
        self.answered_correctly = compare_answers(answer, correct_answer)
        if self.config.eval_method == "llm" and not self.answered_correctly:
            feedback, judgment, reasoning = await evaluate_answer_async(correct_answer, answer, self.current_question["question"], model_config)
            self.answered_correctly = (judgment == "Yes")
            # print(f"💭 Feedback: {feedback}")
            # print(f"💭 Judgment: {judgment}")
            # print(f"💭 Reasoning: {reasoning}")
        
        if self.answered_correctly:
            feedback = f"Correct! Your answer '{answer}' is correct."
            reward = self.config.correct_answer_reward
        else:
            feedback = f"Incorrect. Your answer '{answer}' is not correct."
            reward = self.config.incorrect_answer_reward
        
        if self.config.verbose:
            print(f"💭 Submitted answer: {answer}")
            print(f"✅ Correct!" if self.answered_correctly else f"❌ Incorrect")
        
        return feedback, reward 