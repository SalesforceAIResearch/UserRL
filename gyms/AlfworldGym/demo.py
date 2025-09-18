#!/usr/bin/env python3
"""
Demo script for AlfworldGym.

This script demonstrates how to use the AlfworldGym environment
and serves as a basic functionality test.
"""

def main():
    try:
        print("🚀 AlfworldGym Demo Starting...")
        
        # Import alfworldgym
        from alfworldgym import AlfworldEnv, get_demo_config
        print("✅ AlfworldGym imported successfully")
        
        # Create environment
        config = get_demo_config()
        config.max_steps = 10  # Short demo
        env = AlfworldEnv(config)
        print("✅ Environment created")
        
        # Reset environment
        observation, info = env.reset()
        print(f"\n📋 New Task: {observation['task']}")
        print(f"📍 Initial Feedback: {observation['feedback'][:200]}...")
        
        # Demonstrate some actions
        actions = [
            "[action] look",
            "[action] inventory", 
            "[action] go to kitchen",
            "[finish]"
        ]
        
        for i, action in enumerate(actions, 1):
            print(f"\n🎮 Step {i}: {action}")
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"   💰 Reward: {reward}")
            print(f"   🎯 Terminated: {terminated}, Truncated: {truncated}")
            print(f"   📝 Feedback: {obs['feedback'][:150]}...")
            
            if terminated or truncated:
                print("   🏁 Episode ended")
                break
        
        # Show final render
        print("\n🖼️ Final State:")
        env.render()
        
        # Close environment
        env.close()
        print("\n✅ Demo completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure alfworld is installed and configured correctly.")
        print("Run: pip install alfworld")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 