import json
import re
from typing import Dict, Any, List, Tuple, Optional


def parse_output_as_json(response_text: str) -> Dict[str, Any]:
    """
    Parse the model's response text and extract JSON from it.
    
    Args:
        response_text: Raw response text from the model
        
    Returns:
        Parsed JSON dictionary, or None if parsing fails
    """
    try:
        # Try to parse the entire response as JSON
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        # Look for JSON blocks in the response
        json_pattern = r'```json\s*(.*?)\s*```'
        matches = re.findall(json_pattern, response_text, re.DOTALL)
        
        if matches:
            try:
                return json.loads(matches[0].strip())
            except json.JSONDecodeError:
                pass
        
        # Look for JSON-like content between braces
        brace_pattern = r'\{.*\}'
        matches = re.findall(brace_pattern, response_text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
        
        # If all parsing attempts fail, return None
        return None


def build_history_into_prompt(conversation_history: List[Dict[str, str]], with_note: bool = False) -> str:
    """
    Convert conversation history into a formatted prompt string.
    
    Args:
        conversation_history: List of conversation entries with 'role' and 'content' keys
        with_note: Whether to include notes in the conversation history
        
    Returns:
        Formatted conversation history string
    """
    if not conversation_history:
        return "No previous conversation."
    
    history_str = ""
    for entry in conversation_history:
        role = entry.get("role", "unknown")
        content = entry.get("content", "")
        note = entry.get("note", "") if with_note else ""
        
        if role == "agent":
            history_str += f"Agent: {content}\n"
            if with_note and note:
                history_str += f"[Note: {note}]\n"
        elif role == "user":
            history_str += f"User: {content}\n"
        elif role == "database":
            history_str += f"Database: {content}\n"
        else:
            history_str += f"{role.capitalize()}: {content}\n"
        
        history_str += "\n"
    
    return history_str.strip()


# System and User prompts for judging search requests
JUDGE_SEARCH_SYSTEM = """## **Task**
You are an expert judge evaluating whether an agent's search request aligns with the ground truth search arguments for a travel planning scenario.

## **Instruction**
1. Analyze the agent's search request to determine if it matches any of the ground truth arguments
2. Check if the search request is properly formatted and contains all the relevant information in the ground truth arguments
3. Determine the alignment judgement and identify the specific aspect if aligned
4. Provide your assessment in the specified JSON format

## **Example Format**
```json
{
    "alignment_judgement": "True/False",
    "alignment_aspect": "flight/hotel/restaurant/apartment/rental_car"
}
```

## **Important Notes**
- "True": The search request aligns with one of the ground truth arguments. Note that all the arguments must be covered and correctly covered.
- "False": The search request is malformed or contains incorrect arguments. Note that missing one argument or giving wrong arguments should all be marked as "False".
- For "True" judgements, you must specify the alignment_aspect
- Aspect names should be: flight, hotel, restaurant, apartment, rental_car
- Be strict in your evaluation: Mark false if the request is ambiguous, unclear, or contain multiple search requests. Only mark as "True" if there's clear alignment with all the arguments details covered."""

JUDGE_SEARCH_USER = """**Agent's Search Request:**
{agent_request}

**Ground Truth Arguments:**
{ground_truth_arguments}

Please evaluate the alignment between the agent's search request and the ground truth arguments, then provide your assessment in JSON format wrapped in ```json and ```."""


# System and User prompts for judging agent responses
JUDGE_PROMPT_SYSTEM = """## **Task**
You are an expert judge evaluating the type of an agent's conversation utterance in a travel planning scenario to determine the appropriate response strategy.

## **Instruction**
1. Analyze the agent's latest utterance in the context of the conversation
2. Determine if the agent is explicitly asking for preferences that you have, asking for preferences that you don't have, giving a too general query, or just making general conversations
3. If asking for preferences that you have, identify which specific preference from the available list matches
4. Classify the utterance type and provide the assessment in JSON format

## **Example Format**
```json
{
    "type": "1/2/3/4",
    "preference_id": "preference id if type is 2"
}
```

## **Important Notes**
- Type "1": Normal conversation, not preference-related
- Type "2": Agent explicitly and concretely asking for a preference that exists in the available preferences list. The way how agent asks must be concrete in order to be classified as Type "2".
- Type "3": Agent explicitly and concretely asking for preferences, but the specific preference is not available. Similarly, the way how agent asks must also be concrete and specific.
- Type "4": Agent making a very vague and general query about preference instead of focusing on a specific aspect (e.g. "Do you have any preferences for the car? (vague and general, type 4)" instead of "what exact model of the car do you like? (concrete and specific, type 2)")
- For Type "2", you must provide the exact one preference_id from the available preferences. If there's multiple preferences that match, choose the one that is most relevant to the conversation context.
- Be precise in identifying preference requests vs general conversation"""

JUDGE_PROMPT_USER = """**Travel Scenario:**
{scenario}

**Conversation History:**
{conversation_history}

**Agent's Latest Utterance:**
{latest_utterance}

**Available Preferences:**
{preferences_list}

Please analyze the agent's latest utterance and classify its type, then provide your assessment in JSON format wrapped in ```json and ```."""


# System and User prompts for preference-based responses
RESPONSE_PREFERENCE_SYSTEM = """## **Task**
You are a helpful user in a travel planning conversation who needs to respond to an agent's explicit request for your preference, which you should elicit in an implicit and indirect manner.

## **Instruction**
1. The agent has explicitly asked about a specific preference that you have
2. Respond in a natural, conversational way that reveals your preference implicitly and indirectly
3. Use the provided implicit elicitation statement as guidance, but make it sound natural in context
4. Keep the conversation flowing while sharing your preference information
5. Provide your response in the specified JSON format

## **Example Format**
```json
{   
    "thought": "Your thought process of how to respond naturally and implicitly reveal the preference under the guidance of the implicit elicitation statement",
    "response": "Your natural conversational response that implicitly reveals the preference"
}
```

## **Important Notes**
- Respond naturally as if you're a real person sharing preferences
- Don't directly state "My preference is..." - be more subtle, conversational, and indirect
- Use the implicit elicitation statement as inspiration but adapt it to the conversation context
- Keep responses appropriate length for natural conversation
- Maintain consistency with the conversation history"""

RESPONSE_PREFERENCE_USER = """**Your Preference:**
{preference}

**Conversation History:**
{conversation_history}

**Agent's Latest Utterance:**
{latest_utterance}

Please respond naturally to the agent's request while implicitly sharing your preference under the guidance of the implicit elicitation statement. Provide your response in JSON format wrapped in ```json and ```."""


# System and User prompts for proactive preference elicitation
RESPONSE_ELICIT_SYSTEM = """## **Task**
You are a helpful user in a travel planning conversation who needs to proactively, naturally, but indirectly introduce a preference into the conversation.

## **Instruction**
1. The conversation has gone several turns without preference discussion
2. Naturally steer the conversation to reveal one of your preferences
3. Use the provided implicit elicitation statement as guidance for how to reveal the preference
4. Make the preference revelation feel organic and contextually appropriate, but still in an implicit and indirect manner
5. Provide your response in the specified JSON format

## **Example Format**
```json
{
    "thought": "Your thought process of how to naturally and implicitly introduce the preference under the guidance of the implicit elicitation statement",
    "response": "Your natural conversational response that proactively introduces the preference"
}
```

## **Important Notes**
- Connect to the current conversation context when possible
- Make the preference introduction feel natural and not forced, but still in an implicit and indirect manner
- Use the implicit elicitation statement as inspiration but adapt to the conversation flow
- Don't abruptly change topics - find natural transitions and keep responses conversational and engaging
- If the implicit elicitation statement cannot clearly what high-level aspect (flight, restaurant, etc.) the preference is about, you should be clear about the high-level aspect in your elicitation to avoid confusion, but still elicit the concrete preference in an implicit way"""

RESPONSE_ELICIT_USER = """**Preference to Elicit:**
{preference}

**Conversation History:**
{conversation_history}

**Agent's Latest Utterance:**
{latest_utterance}

Please respond naturally while proactively introducing your preference into the conversation in an implicit and indirect manner under the guidance of the implicit elicitation statement. Provide your response in JSON format wrapped in ```json and ```."""


# System and User prompts for natural conversation responses
RESPONSE_NATURAL_SYSTEM = """## **Task**
You are a helpful user in a travel planning conversation who needs to respond naturally to the agent's utterance.

## **Instruction**
1. The agent's utterance is not related to any specific preferences you have
2. Respond naturally and in a succinct manner, like you are giving a half-hearted reply
3. Be neutral and do not reveal any new or arbitrary personal preferences
4. Provide your response in the specified JSON format

## **Example Format**
```json
{
    "thought": "Your thought process of how to respond naturally and keep the conversation flowing while being neutral",
    "response": "Your natural conversational response"
}
```

## **Important Notes**
- Keep responses natural, conversational and succinct
- Stay on topic with travel planning when appropriate, but do not actively ask any questions
- Don't introduce any personal preferences. If being asked, you should be neutral (e.g. "I don't have a preference on that", "Everything is fine") and do not arbitrarily reveal any new preferences.
"""

RESPONSE_NATURAL_USER = """**Conversation History:**
{conversation_history}

**Agent's Latest Utterance:**
{latest_utterance}

Please respond naturally to continue the conversation and keep neutral without revealing any new or arbitrary personal preferences. Provide your response in JSON format wrapped in ```json and ```."""


OPTION_SCHEMAS = {
    "flight": """Each flight option includes:
    - path: Array of airport/city names from origin to destination, including layovers
    - time: Flight and layover durations in hours (alternating: flight, layover, flight, etc.)
    - company: Airline names for each leg of the trip
    - flight_number: Flight numbers corresponding to each airline leg
    - cost: Total ticket price in USD for economy class
    - amenities: Available in-flight amenities free of charge
    - service: Additional services with costs
    """,
    "hotel": """Each hotel option includes:
    - name: Hotel name
    - room: Array of available room types with capacity
    - cost: Array of costs for different room types for the entire stay in USD
    - rating: Hotel rating on a scale of 0-10
    - amenities: Available facilities free of charge
    - service: Additional services with costs
    """,
    "restaurant": """Each restaurant option includes:
    - name: Restaurant name
    - cuisine: Type of cuisine
    - expectation: Price level expectation
    - rating: Average restaurant rating on a scale of 0-10
    - reviews: Breakdown of customer reviews by star rating
    - tags: Descriptive features
    """,
    "apartment": """Each apartment rental option includes:
    - name: Apartment listing name or brand
    - room: Room configuration using bedroom/bathroom format (1B1B, 2B1B, 2B2B, 3B2B, 3B3B, 4B4B, 6B6B, 8B8B)
    - capacity: Maximum number of guests the apartment can accommodate
    - cost: Total rental cost for the entire stay in USD
    - rating: Apartment rating on a scale of 0-10
    - amenities: Available facilities free of charge
    - service: Additional services with costs
    """,
    "rental_car": """Each rental car option includes:
    - brand: Car rental company name
    - model: Specific car model
    - categories: Vehicle category
    - seats: Number of passenger seats
    - cost: Total rental cost for the entire period in USD
    - insurance: Available insurance options with costs
    - service: Additional services with costs
    """
}