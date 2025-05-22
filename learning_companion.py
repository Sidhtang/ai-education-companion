import google.generativeai as genai
import gradio as gr
import numpy as np
import pandas as pd
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, Counter
import random
from dotenv import load_dotenv
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("learning_companion.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables (for API keys)
load_dotenv()

# =============== 1. ENHANCED LLM INTEGRATION ===============

class ContentGenerator:
    """Advanced content generation with multiple LLM options and caching"""

    def __init__(self):
        self.setup_gemini()
        self.content_cache = {}
        self.fallback_enabled = True

    def setup_gemini(self):
        """Setup Gemini API with safe error handling"""
        try:
            # Hardcode the API key here
            api_key = os.getenv("GEMINI_API_KEY")
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
            logger.info("Gemini API initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {str(e)}")
            self.gemini_model = None

    def _get_cache_key(self, topic, difficulty, style, content_type):
        """Create a unique cache key"""
        return f"{topic}|{difficulty}|{style}|{content_type}"

    def generate_content(self, topic, difficulty, style, content_type="full"):
        """Generate educational content with caching and improved prompting"""
        # Check cache first
        cache_key = self._get_cache_key(topic, difficulty, style, content_type)
        if cache_key in self.content_cache:
            logger.info(f"Cache hit for {cache_key}")
            return self.content_cache[cache_key]

        # Style-specific instructions
        style_instructions = {
            "visual": "Include diagrams, charts, and visual representations. Use descriptive language that evokes mental imagery. Format key concepts in tables when appropriate.",
            "auditory": "Structure content as a conversational dialogue. Include mnemonics, rhythmic patterns, and verbal cues. Emphasize verbal explanations.",
            "kinesthetic": "Design hands-on experiments or activities. Include interactive elements where students can apply concepts immediately. Suggest real-world applications.",
            "reading/writing": "Provide well-structured text with clear headings. Include detailed written explanations and opportunities for note-taking.",
            "multimodal": "Blend multiple learning modalities. Combine visual elements with verbal explanations and suggested activities."
        }

        # Difficulty-specific adaptations
        complexity_adjustments = {
            "beginner": "Use simple language and basic concepts. Avoid jargon and explain all technical terms. Focus on foundational knowledge.",
            "intermediate": "Build on basic concepts with moderate complexity. Introduce more technical vocabulary. Connect concepts to create a broader understanding.",
            "advanced": "Use sophisticated language and complex concepts. Explore nuanced aspects of the topic. Challenge students with deeper analysis."
        }

        # Content type specific format
        format_instructions = {
            "full": """
Structure your response in these clear sections:
1. CONCEPT EXPLANATION: Provide a clear, concise explanation of the core concept.
2. KEY POINTS: List 3-5 essential takeaways in bullet points.
3. DETAILED EXAMPLE: Walk through a comprehensive example step-by-step.
4. PRACTICE PROBLEM: Create a challenging but appropriate practice problem.
5. SOLUTION: Provide a detailed solution with explanation.
6. FURTHER EXPLORATION: Suggest ways to deepen understanding of this topic.
            """,
            "lesson": """
Structure your response in these sections:
1. CONCEPT EXPLANATION: Provide a clear, concise explanation of the core concept.
2. KEY POINTS: List 3-5 essential takeaways in bullet points.
3. DETAILED EXAMPLE: Walk through a comprehensive example step-by-step.
            """,
            "quiz": """
Create exactly one practice problem related to this topic, appropriate for the specified difficulty level.
Then provide a detailed solution with explanation, showing all steps clearly.
Label the sections as:
- PRACTICE PROBLEM
- SOLUTION
            """
        }

        # Generate enhanced prompt
        prompt = f"""
You are an expert educational content creator specializing in personalized learning.

TOPIC: {topic}
DIFFICULTY: {difficulty}
LEARNING STYLE: {style}

SPECIFIC STYLE GUIDANCE: {style_instructions.get(style, style_instructions["multimodal"])}
DIFFICULTY ADAPTATION: {complexity_adjustments.get(difficulty, complexity_adjustments["intermediate"])}

{format_instructions.get(content_type, format_instructions["full"])}

Make all content highly engaging and relevant to real-world applications when possible.
        """

        # Generate content with error handling
        try:
            if self.gemini_model:
                response = self.gemini_model.generate_content(prompt)
                content = response.text
                # Cache the result
                self.content_cache[cache_key] = content
                return content
            else:
                return self._generate_fallback_content(topic, difficulty, style, content_type)
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            return self._generate_fallback_content(topic, difficulty, style, content_type)

    def _generate_fallback_content(self, topic, difficulty, style, content_type):
        """Generate fallback content when API fails"""
        if content_type == "quiz":
            return f"""
PRACTICE PROBLEM:
Create a brief explanation of {topic} at a {difficulty} level, considering {style} learning preferences.

SOLUTION:
A good explanation would include key concepts, examples, and applications relevant to {topic}.
            """
        else:
            return f"""
# Learning Content: {topic}

Sorry, I'm currently having trouble connecting to the content generation service.

Please try again later or explore these resources about {topic}:
- Check educational websites like Khan Academy or Coursera
- Look for {topic} in textbooks or academic journals
- Try searching for "{topic} tutorial" online

This is a temporary technical issue, and we appreciate your patience.
            """

# =============== 2. ADVANCED REINFORCEMENT LEARNING SYSTEM ===============

class LearningState:
    """Comprehensive student learning state representation"""

    def __init__(self):
        # Make sure we have exactly 13 features as expected by the network
        self.features = {
            # Performance metrics
            'correct_answers': 0,
            'incorrect_answers': 0,
            'consecutive_correct': 0,
            'total_attempts': 0,
            'avg_response_time': 0,

            # Current settings
            'difficulty_level': 0,  # 0=beginner, 1=intermediate, 2=advanced
            'learning_style': 0,    # 0=visual, 1=auditory, 2=kinesthetic, 3=reading/writing, 4=multimodal

            # Engagement metrics
            'session_duration': 0,
            'topics_explored': 0,

            # Temporal features
            'time_since_last_activity': 0,
            'day_of_week': datetime.now().weekday(),
            'time_of_day': datetime.now().hour,
            'engagement_score': 0,  # Added to ensure 13 features
        }

        self.difficulty_map = ['beginner', 'intermediate', 'advanced']
        self.style_map = ['visual', 'auditory', 'kinesthetic', 'reading/writing', 'multimodal']
        self.history = []
        self.topic_history = Counter()
        self.last_activity_time = datetime.now()

    def update(self, correct, response_time, current_topic):
        """Update state based on student interaction"""
        # Record previous state for history
        self.history.append(self.features.copy())

        # Update performance metrics
        self.features['total_attempts'] += 1
        if correct:
            self.features['correct_answers'] += 1
            self.features['consecutive_correct'] += 1
        else:
            self.features['incorrect_answers'] += 1
            self.features['consecutive_correct'] = 0

        # Update response time metrics
        prev_avg = self.features['avg_response_time']
        n = self.features['total_attempts']
        self.features['avg_response_time'] = (prev_avg * (n-1) + response_time) / n

        # Update engagement metrics
        now = datetime.now()
        time_diff = (now - self.last_activity_time).total_seconds()
        self.features['time_since_last_activity'] = time_diff
        self.last_activity_time = now

        # Update session metrics
        self.features['session_duration'] += time_diff

        # Update topic tracking
        if current_topic not in self.topic_history:
            self.features['topics_explored'] += 1
        self.topic_history[current_topic] += 1

        # Update temporal features
        self.features['day_of_week'] = now.weekday()
        self.features['time_of_day'] = now.hour

        return self.features.copy()

    def get_current_difficulty(self):
        """Get current difficulty setting as string"""
        idx = min(int(self.features['difficulty_level']), len(self.difficulty_map)-1)
        return self.difficulty_map[idx]

    def get_current_style(self):
        """Get current learning style setting as string"""
        idx = min(int(self.features['learning_style']), len(self.style_map)-1)
        return self.style_map[idx]

    def set_difficulty(self, difficulty):
        """Set difficulty by name"""
        if difficulty in self.difficulty_map:
            self.features['difficulty_level'] = self.difficulty_map.index(difficulty)
        else:
            self.features['difficulty_level'] = 0

    def set_style(self, style):
        """Set learning style by name"""
        if style in self.style_map:
            self.features['learning_style'] = self.style_map.index(style)
        else:
            self.features['learning_style'] = 0

    def get_state_tensor(self):
        """Convert state to tensor for RL model - ensure correct shape"""
        # Ensure we're using exactly the features expected by the network
        feature_values = list(self.features.values())
        # Double-check we have the right number of features
        assert len(feature_values) == 13, f"Expected 13 features, got {len(feature_values)}"
        return torch.tensor([feature_values], dtype=torch.float32)

    def get_performance_ratio(self):
        """Calculate performance ratio for adaptive difficulty"""
        total = self.features['correct_answers'] + self.features['incorrect_answers']
        if total == 0:
            return 0.5
        return self.features['correct_answers'] / total

class DQNetwork(nn.Module):
    """Deep Q-Network with advanced architecture"""

    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()

        # Larger network with dropout for regularization
        self.fc1 = nn.Linear(state_size, 128)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)

        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class ReplayBuffer:
    """Experience replay buffer with prioritized sampling"""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = 0.6  # Priority exponent
        self.epsilon = 0.01  # Small constant to avoid zero priority

    def add(self, state, action, reward, next_state, done, error=None):
        """Add experience to buffer with priority"""
        experience = (state, action, reward, next_state, done)

        # Set initial priority based on TD error or default to max priority
        if error is None:
            priority = max(self.priorities) if self.priorities else 1.0
        else:
            priority = abs(error) + self.epsilon

        self.buffer.append(experience)
        self.priorities.append(priority)

    def sample(self, batch_size):
        """Sample batch with prioritized experience replay"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        # Convert priorities to probabilities
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        # Extract experiences
        states = torch.cat([self.buffer[i][0] for i in indices])
        actions = torch.tensor([self.buffer[i][1] for i in indices], dtype=torch.long)
        rewards = torch.tensor([self.buffer[i][2] for i in indices], dtype=torch.float32)
        next_states = torch.cat([self.buffer[i][3] for i in indices])
        dones = torch.tensor([self.buffer[i][4] for i in indices], dtype=torch.bool)

        return states, actions, rewards, next_states, dones, indices

    def update_priorities(self, indices, errors):
        """Update priorities based on new TD errors"""
        for i, error in zip(indices, errors):
            if i < len(self.priorities):  # Safety check
                self.priorities[i] = abs(error) + self.epsilon

    def __len__(self):
        return len(self.buffer)

class AdaptiveLearningAgent:
    """Advanced Reinforcement Learning agent for adaptive learning"""

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Double-check that state_size matches our feature count
        if state_size != 13:
            logger.warning(f"Expected state_size to be 13, got {state_size}. Adjusting to 13.")
            self.state_size = 13

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Networks
        self.policy_net = DQNetwork(self.state_size, action_size).to(self.device)
        self.target_net = DQNetwork(self.state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network in evaluation mode

        # Optimizer with learning rate scheduler
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)

        # Replay buffer
        self.memory = ReplayBuffer(capacity=50000)

        # Exploration parameters with annealing
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.gamma = 0.99  # Discount factor

        # Training parameters
        self.batch_size = 64
        self.target_update = 10  # Update target network every N steps
        self.steps_done = 0

        # Performance tracking
        self.rewards_history = []
        self.loss_history = []

        # Action mapping for interpretability
        self.action_map = self._create_action_map()

        # Load model if available
        self.model_path = "learning_companion_model.pth"
        self.load_model()

    def _create_action_map(self):
        """Create mapping between action indices and actual changes"""
        actions = []
        difficulty_changes = [-1, 0, 1]  # Decrease, maintain, increase
        style_changes = [-1, 0, 1]      # Previous, maintain, next style

        for d_change in difficulty_changes:
            for s_change in style_changes:
                actions.append({
                    'difficulty_change': d_change,
                    'style_change': s_change
                })

        return actions

    def select_action(self, state):
        """Select action using epsilon-greedy policy with decaying exploration"""
        if random.random() < self.epsilon:
            action = random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state_tensor = state.to(self.device)
                q_values = self.policy_net(state_tensor)
                action = q_values.max(1)[1].item()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.steps_done += 1
        return action

    def apply_action(self, action_idx, learning_state):
        """Apply selected action to learning state"""
        action = self.action_map[action_idx]

        # Get current settings
        current_difficulty = learning_state.features['difficulty_level']
        current_style = learning_state.features['learning_style']

        # Apply difficulty change
        new_difficulty = current_difficulty + action['difficulty_change']
        new_difficulty = max(0, min(new_difficulty, len(learning_state.difficulty_map) - 1))

        # Apply style change
        new_style = current_style + action['style_change']
        new_style = max(0, min(new_style, len(learning_state.style_map) - 1))

        # Update state
        learning_state.features['difficulty_level'] = new_difficulty
        learning_state.features['learning_style'] = new_style

        return learning_state

    def calculate_reward(self, correct, consecutive_correct, difficulty_level):
        """Calculate more nuanced reward based on performance and context"""
        # Base reward for correctness
        base_reward = 1.0 if correct else -0.5

        # Bonus for consecutive correct answers
        streak_bonus = min(consecutive_correct * 0.1, 0.5) if correct else 0

        # Difficulty adjustment
        difficulty_factor = 1.0 + (difficulty_level * 0.25)

        # Combine rewards
        reward = (base_reward + streak_bonus) * difficulty_factor

        return reward

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay memory"""
        # Calculate TD error for prioritized replay
        with torch.no_grad():
            state_tensor = state.to(self.device)
            next_state_tensor = next_state.to(self.device)

            current_q = self.policy_net(state_tensor)[0][action].item()
            next_q = self.target_net(next_state_tensor).max(1)[0].item()

            expected_q = reward + self.gamma * next_q * (1 - int(done))
            td_error = expected_q - current_q

        self.memory.add(state, action, reward, next_state, done, td_error)
        self.rewards_history.append(reward)

    def optimize_model(self):
        """Perform one step of optimization"""
        if len(self.memory) < self.batch_size:
            return 0.0  # Not enough samples

        # Sample batch with priorities
        states, actions, rewards, next_states, dones, indices = self.memory.sample(self.batch_size)

        # Move tensors to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Compute Q values
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        expected_q_values = rewards + self.gamma * next_q_values * (~dones)

        # Compute TD errors for priority update
        td_errors = expected_q_values - q_values.squeeze()

        # Huber loss for stability
        criterion = nn.SmoothL1Loss()
        loss = criterion(q_values.squeeze(), expected_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update priorities
        self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())

        # Update target network
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Update learning rate
        self.scheduler.step()

        # Record loss
        loss_value = loss.item()
        self.loss_history.append(loss_value)

        return loss_value

    def save_model(self):
        """Save model weights"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'rewards_history': self.rewards_history,
            'loss_history': self.loss_history
        }, self.model_path)
        logger.info(f"Model saved to {self.model_path}")

    def load_model(self):
        """Load model weights if available"""
        try:
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path)
                self.policy_net.load_state_dict(checkpoint['policy_net'])
                self.target_net.load_state_dict(checkpoint['target_net'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                self.epsilon = checkpoint['epsilon']
                self.steps_done = checkpoint['steps_done']
                self.rewards_history = checkpoint['rewards_history']
                self.loss_history = checkpoint['loss_history']
                logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")

    def plot_learning_curves(self):
        """Plot learning curves for visualization"""
        plt.figure(figsize=(12, 8))

        # Plot rewards
        plt.subplot(2, 1, 1)
        rewards = pd.Series(self.rewards_history).rolling(100).mean()
        plt.plot(rewards)
        plt.title('Average Reward (100-episode rolling mean)')
        plt.xlabel('Episodes')
        plt.ylabel('Reward')

        # Plot loss
        plt.subplot(2, 1, 2)
        loss = pd.Series(self.loss_history).rolling(100).mean()
        plt.plot(loss)
        plt.title('Average Loss (100-episode rolling mean)')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')

        plt.tight_layout()
        plt.savefig('visualizations/learning_curves.png')
        plt.close()

# =============== 3. LEARNING ANALYTICS & STUDENT PROFILING ===============

class LearnerProfile:
    """Student profiling and analytics system"""

    def __init__(self, user_id="default_user"):
        self.user_id = user_id
        self.data_file = f"user_profiles/{user_id}.json"
        self.current_session = {
            "start_time": datetime.now().isoformat(),
            "interactions": [],
            "topics": {}
        }
        self.profile = self._load_profile()

    def _load_profile(self):
        """Load existing profile or create new one"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    return json.load(f)
            else:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        except Exception as e:
            logger.error(f"Error loading profile: {str(e)}")

        # Default profile
        return {
            "user_id": self.user_id,
            "created_at": datetime.now().isoformat(),
            "learning_preferences": {
                "preferred_style": None,
                "optimal_difficulty": None,
                "topic_interests": {},
                "engagement_patterns": {
                    "average_session_duration": 0,
                    "peak_activity_times": []
                }
            },
            "performance_metrics": {
                "topics": {},
                "overall": {
                    "correct_answers": 0,
                    "incorrect_answers": 0,
                    "total_attempts": 0
                }
            },
            "sessions": []
        }

    def record_interaction(self, interaction_data):
        """Record student interaction"""
        # Add timestamp
        interaction_data["timestamp"] = datetime.now().isoformat()
        self.current_session["interactions"].append(interaction_data)

        # Update topic tracking
        topic = interaction_data.get("topic", "unknown")
        if topic not in self.current_session["topics"]:
            self.current_session["topics"][topic] = {
                "attempts": 0,
                "correct": 0
            }

        self.current_session["topics"][topic]["attempts"] += 1
        if interaction_data.get("is_correct", False):
            self.current_session["topics"][topic]["correct"] += 1

        # Update overall metrics
        if "performance_metrics" not in self.profile:
            self.profile["performance_metrics"] = {"topics": {}, "overall": {"correct_answers": 0, "incorrect_answers": 0, "total_attempts": 0}}

        self.profile["performance_metrics"]["overall"]["total_attempts"] += 1
        if interaction_data.get("is_correct", False):
            self.profile["performance_metrics"]["overall"]["correct_answers"] += 1
        else:
            self.profile["performance_metrics"]["overall"]["incorrect_answers"] += 1

        # Update topic-specific metrics
        if topic not in self.profile["performance_metrics"]["topics"]:
            self.profile["performance_metrics"]["topics"][topic] = {
                "attempts": 0,
                "correct": 0,
                "mastery_level": 0.0
            }

        self.profile["performance_metrics"]["topics"][topic]["attempts"] += 1
        if interaction_data.get("is_correct", False):
            self.profile["performance_metrics"]["topics"][topic]["correct"] += 1

        # Calculate mastery level (simple version)
        attempts = self.profile["performance_metrics"]["topics"][topic]["attempts"]
        correct = self.profile["performance_metrics"]["topics"][topic]["correct"]
        mastery = correct / max(attempts, 1) * 100
        self.profile["performance_metrics"]["topics"][topic]["mastery_level"] = mastery

        # Save profile
        self._save_profile()

    def end_session(self):
        """End current session and save data"""
        # Calculate session duration
        start_time = datetime.fromisoformat(self.current_session["start_time"])
        end_time = datetime.now()
        duration_seconds = (end_time - start_time).total_seconds()
        self.current_session["duration_seconds"] = duration_seconds
        self.current_session["end_time"] = end_time.isoformat()

        # Add session to profile
        if "sessions" not in self.profile:
            self.profile["sessions"] = []
        self.profile["sessions"].append(self.current_session)

        # Update engagement patterns
        if "learning_preferences" not in self.profile:
            self.profile["learning_preferences"] = {
                "preferred_style": None,
                "optimal_difficulty": None,
                "topic_interests": {},
                "engagement_patterns": {
                    "average_session_duration": 0,
                    "peak_activity_times": []
                }
            }

        # Update average session duration
        session_durations = [session.get("duration_seconds", 0) for session in self.profile["sessions"]]
        self.profile["learning_preferences"]["engagement_patterns"]["average_session_duration"] = sum(session_durations) / max(len(session_durations), 1)

        # Update peak activity times
        hour_of_day = start_time.hour
        if hour_of_day not in self.profile["learning_preferences"]["engagement_patterns"]["peak_activity_times"]:
            self.profile["learning_preferences"]["engagement_patterns"]["peak_activity_times"].append(hour_of_day)

        # Update topic interests
        for topic, data in self.current_session["topics"].items():
            if "topic_interests" not in self.profile["learning_preferences"]:
                self.profile["learning_preferences"]["topic_interests"] = {}

            if topic not in self.profile["learning_preferences"]["topic_interests"]:
                self.profile["learning_preferences"]["topic_interests"][topic] = 0

            self.profile["learning_preferences"]["topic_interests"][topic] += data["attempts"]

        # Analyze learning style preference
        self._analyze_learning_preferences()

        # Save profile
        self._save_profile()

        # Reset current session
        self.current_session = {
            "start_time": datetime.now().isoformat(),
            "interactions": [],
            "topics": {}
        }

    def _analyze_learning_preferences(self):
        """Analyze student data to determine learning preferences"""
        # Count style effectiveness
        style_performance = {}
        difficulty_performance = {}

        for session in self.profile["sessions"]:
            for interaction in session["interactions"]:
                style = interaction.get("learning_style")
                difficulty = interaction.get("difficulty")
                is_correct = interaction.get("is_correct", False)
                if style and difficulty:
                    # Initialize counters if needed
                    if style not in style_performance:
                        style_performance[style] = {"correct": 0, "total": 0}
                    if difficulty not in difficulty_performance:
                        difficulty_performance[difficulty] = {"correct": 0, "total": 0}

                    # Update counters
                    style_performance[style]["total"] += 1
                    difficulty_performance[difficulty]["total"] += 1

                    if is_correct:
                        style_performance[style]["correct"] += 1
                        difficulty_performance[difficulty]["correct"] += 1

        # Determine preferred style
        best_style = None
        best_style_ratio = 0

        for style, data in style_performance.items():
            if data["total"] >= 5:  # Minimum threshold for confident assessment
                ratio = data["correct"] / data["total"]
                if ratio > best_style_ratio:
                    best_style_ratio = ratio
                    best_style = style

        # Determine optimal difficulty
        best_difficulty = None
        best_difficulty_ratio = 0

        for difficulty, data in difficulty_performance.items():
            if data["total"] >= 5:  # Minimum threshold for confident assessment
                ratio = data["correct"] / data["total"]
                # Ideal ratio around 0.7-0.8 (challenging but achievable)
                adjusted_ratio = 1 - abs(0.75 - ratio)
                if adjusted_ratio > best_difficulty_ratio:
                    best_difficulty_ratio = adjusted_ratio
                    best_difficulty = difficulty

        # Update profile
        if best_style:
            self.profile["learning_preferences"]["preferred_style"] = best_style

        if best_difficulty:
            self.profile["learning_preferences"]["optimal_difficulty"] = best_difficulty

    def _save_profile(self):
        """Save user profile to file"""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.profile, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving profile: {str(e)}")

    def get_learning_insights(self):
        """Generate insights from learner profile"""
        insights = {
            "strengths": [],
            "areas_for_improvement": [],
            "recommendations": [],
            "learning_style": self.profile["learning_preferences"]["preferred_style"],
            "optimal_difficulty": self.profile["learning_preferences"]["optimal_difficulty"]
        }

        # Find strengths (topics with high mastery)
        for topic, data in self.profile["performance_metrics"]["topics"].items():
            if data["attempts"] >= 3:  # Only consider topics with enough attempts
                if data["mastery_level"] >= 80:
                    insights["strengths"].append({
                        "topic": topic,
                        "mastery": data["mastery_level"]
                    })
                elif data["mastery_level"] <= 50:
                    insights["areas_for_improvement"].append({
                        "topic": topic,
                        "mastery": data["mastery_level"]
                    })

        # Sort by mastery level
        insights["strengths"].sort(key=lambda x: x["mastery"], reverse=True)
        insights["areas_for_improvement"].sort(key=lambda x: x["mastery"])

        # Limit to top 3
        insights["strengths"] = insights["strengths"][:3]
        insights["areas_for_improvement"] = insights["areas_for_improvement"][:3]

        # Generate recommendations
        if insights["areas_for_improvement"]:
            for area in insights["areas_for_improvement"]:
                insights["recommendations"].append({
                    "topic": area["topic"],
                    "suggestion": f"Review {area['topic']} with {insights['learning_style'] or 'multimodal'} resources at {insights['optimal_difficulty'] or 'intermediate'} level"
                })

        # Add general recommendation if no specific areas found
        if not insights["recommendations"]:
            top_interests = sorted(
                self.profile["learning_preferences"]["topic_interests"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]

            for topic, _ in top_interests:
                insights["recommendations"].append({
                    "topic": topic,
                    "suggestion": f"Explore advanced concepts in {topic} to deepen understanding"
                })

        return insights

    def generate_progress_report(self):
        """Generate a comprehensive progress report"""
        # Overall statistics
        overall = self.profile["performance_metrics"]["overall"]
        total_attempts = overall["total_attempts"]
        accuracy = (overall["correct_answers"] / total_attempts * 100) if total_attempts > 0 else 0

        # Session statistics
        num_sessions = len(self.profile["sessions"])
        avg_duration = self.profile["learning_preferences"]["engagement_patterns"]["average_session_duration"] / 60  # in minutes

        # Topic mastery
        topic_mastery = []
        for topic, data in self.profile["performance_metrics"]["topics"].items():
            if data["attempts"] >= 3:  # Only consider topics with enough attempts
                topic_mastery.append({
                    "topic": topic,
                    "mastery": data["mastery_level"],
                    "attempts": data["attempts"]
                })

        # Sort by mastery
        topic_mastery.sort(key=lambda x: x["mastery"], reverse=True)

        # Get insights
        insights = self.get_learning_insights()

        return {
            "summary": {
                "total_sessions": num_sessions,
                "total_questions_attempted": total_attempts,
                "overall_accuracy": accuracy,
                "average_session_duration": avg_duration
            },
            "topic_mastery": topic_mastery,
            "learning_preferences": {
                "preferred_style": self.profile["learning_preferences"]["preferred_style"],
                "optimal_difficulty": self.profile["learning_preferences"]["optimal_difficulty"]
            },
            "insights": insights
        }

# =============== 4. COMPREHENSIVE EDUCATIONAL SYSTEM INTEGRATION ===============

class LearningCompanion:
    """Main learning companion system integrating all components"""

    def __init__(self, user_id="default_user"):
        # Initialize components
        self.content_generator = ContentGenerator()
        self.state = LearningState()

        # Initialize learner profile
        self.learner_profile = LearnerProfile(user_id)

        # Initialize RL agent (13 state features, 9 actions)
        self.agent = AdaptiveLearningAgent(13, 9)

        # Current topic and content
        self.current_topic = None
        self.current_content = None
        self.last_action_time = datetime.now()

        logger.info("Learning Companion initialized")

    def generate_content(self, topic, content_type="full"):
        """Generate content for the specified topic"""
        if not topic:
            return "Please specify a topic"

        # Record topic
        self.current_topic = topic

        # Get current difficulty and learning style
        difficulty = self.state.get_current_difficulty()
        learning_style = self.state.get_current_style()

        # Generate content
        content = self.content_generator.generate_content(
            topic,
            difficulty,
            learning_style,
            content_type
        )

        # Store current content
        self.current_content = content

        # Log
        logger.info(f"Generated {content_type} content for '{topic}' at {difficulty} level with {learning_style} style")

        return content

    def evaluate_answer(self, user_answer, is_correct=None):
        """Evaluate user answer and update learning state"""
        if self.current_topic is None:
            return "No current topic. Please generate content first."

        # Calculate response time
        now = datetime.now()
        response_time = (now - self.last_action_time).total_seconds()
        self.last_action_time = now

        # If correct flag not provided, assume it's manually marked
        if is_correct is None:
            return "Please mark the answer as correct or incorrect"

        try:
            # Current state (before update)
            current_state_tensor = self.state.get_state_tensor()
            # Verify shape
            assert current_state_tensor.shape[1] == 13, f"State tensor has wrong shape: {current_state_tensor.shape}"

            # Update state
            self.state.update(is_correct, response_time, self.current_topic)

            # Record interaction in learner profile
            self.learner_profile.record_interaction({
                "topic": self.current_topic,
                "is_correct": is_correct,
                "response_time": response_time,
                "difficulty": self.state.get_current_difficulty(),
                "learning_style": self.state.get_current_style()
            })

            # Calculate reward
            reward = self.agent.calculate_reward(
                is_correct,
                self.state.features['consecutive_correct'],
                self.state.features['difficulty_level']
            )

            # Select action based on updated state
            next_state_tensor = self.state.get_state_tensor()
            # Verify shape
            assert next_state_tensor.shape[1] == 13, f"Next state tensor has wrong shape: {next_state_tensor.shape}"

            action = self.agent.select_action(next_state_tensor)

            # Store transition
            self.agent.store_transition(
                current_state_tensor,
                action,
                reward,
                next_state_tensor,
                False  # Not a terminal state
            )

            # Apply action to learning state
            self.state = self.agent.apply_action(action, self.state)

            # Optimize model
            loss = self.agent.optimize_model()

            # Prepare response
            if is_correct:
                feedback = f"Correct! Your answer demonstrates understanding of {self.current_topic}."
            else:
                feedback = f"Not quite right. Let's review {self.current_topic} again."

            # Add adaptation info
            adaptation = f"\n\nAdjusting to your learning: Now using {self.state.get_current_style()} style at {self.state.get_current_difficulty()} level."

            # Save agent model periodically
            if self.agent.steps_done % 50 == 0:
                self.agent.save_model()

            return feedback + adaptation

        except Exception as e:
            logger.error(f"Error in evaluate_answer: {str(e)}")
            return f"An error occurred while evaluating your answer. Technical details: {str(e)}"

    def get_insights(self):
        """Get learning insights for the student"""
        return self.learner_profile.get_learning_insights()

    def get_progress_report(self):
        """Generate comprehensive progress report"""
        return self.learner_profile.generate_progress_report()

    def end_session(self):
        """End the current learning session"""
        self.learner_profile.end_session()
        self.agent.save_model()
        return "Session ended and progress saved."

    def visualize_learning(self):
        """Generate visualizations of learning progress"""
        self.agent.plot_learning_curves()
        return "Learning curves generated and saved to 'visualizations/learning_curves.png'"

# =============== 5. GRADIO INTERFACE ===============

def create_interface():
    """Create Gradio interface for the Learning Companion"""

    # Initialize learning companion with better error handling
    try:
        companion = LearningCompanion()
        logger.info("Learning Companion initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing Learning Companion: {str(e)}")
        companion = None  # Will handle this in the UI functions

    # Define Gradio components and functions with proper error handling
    def generate_content_fn(topic, content_type):
        """Generate content with better error handling"""
        try:
            if not companion:
                return "Error: Learning Companion not initialized properly. Check logs for details."

            if not topic or topic.strip() == "":
                return "Please enter a topic to generate content."

            result = companion.generate_content(topic, content_type)

            # Verify result is not None or empty
            if not result or result.strip() == "":
                return "Error generating content. Please try a different topic or check logs."

            return result
        except Exception as e:
            error_msg = f"Error generating content: {str(e)}"
            logger.error(error_msg)
            return f"Something went wrong while generating content. Please try again with a different topic.\n\nTechnical details: {str(e)}"

    def evaluate_answer_fn(user_answer, is_correct):
        """Evaluate answer with proper error handling"""
        try:
            if not companion:
                return "Error: Learning Companion not initialized properly. Check logs for details."

            if not user_answer or user_answer.strip() == "":
                return "Please enter an answer before submitting."

            if is_correct is None:
                return "Please mark your answer as correct or incorrect."

            # Convert string to boolean
            is_correct_bool = (is_correct == "Correct")

            result = companion.evaluate_answer(user_answer, is_correct_bool)
            return result
        except Exception as e:
            error_msg = f"Error evaluating answer: {str(e)}"
            logger.error(error_msg)
            return f"Something went wrong while evaluating your answer. Please try again.\n\nTechnical details: {str(e)}"

    def get_insights_fn():
        """Get insights with proper error handling"""
        try:
            if not companion:
                return "Error: Learning Companion not initialized properly. Check logs for details."

            insights = companion.get_insights()

            # Format insights for display
            result = "## Learning Insights\n\n"

            result += "### Strengths\n"
            if insights["strengths"]:
                for strength in insights["strengths"]:
                    result += f"- {strength['topic']}: {strength['mastery']:.1f}% mastery\n"
            else:
                result += "- Not enough data to determine strengths yet.\n"

            result += "\n### Areas for Improvement\n"
            if insights["areas_for_improvement"]:
                for area in insights["areas_for_improvement"]:
                    result += f"- {area['topic']}: {area['mastery']:.1f}% mastery\n"
            else:
                result += "- Not enough data to determine areas for improvement yet.\n"

            result += "\n### Recommendations\n"
            if insights["recommendations"]:
                for rec in insights["recommendations"]:
                    result += f"- {rec['suggestion']}\n"
            else:
                result += "- Continue learning to receive personalized recommendations.\n"

            result += f"\n### Learning Preferences\n"
            result += f"- Preferred Learning Style: {insights['learning_style'] or 'Not enough data'}\n"
            result += f"- Optimal Difficulty Level: {insights['optimal_difficulty'] or 'Not enough data'}\n"

            return result
        except Exception as e:
            error_msg = f"Error getting insights: {str(e)}"
            logger.error(error_msg)
            return f"Something went wrong while generating insights. Please try again later.\n\nTechnical details: {str(e)}"

    def get_progress_report_fn():
        """Get progress report with proper error handling"""
        try:
            if not companion:
                return "Error: Learning Companion not initialized properly. Check logs for details."

            report = companion.get_progress_report()

            # Format report for display
            result = "# Learning Progress Report\n\n"

            # Summary
            summary = report["summary"]
            result += "## Summary\n"
            result += f"- Total Sessions: {summary['total_sessions']}\n"
            result += f"- Questions Attempted: {summary['total_questions_attempted']}\n"
            result += f"- Overall Accuracy: {summary['overall_accuracy']:.1f}%\n"
            result += f"- Average Session Duration: {summary['average_session_duration']:.1f} minutes\n\n"

            # Topic Mastery
            result += "## Topic Mastery\n"
            if report["topic_mastery"]:
                for topic in report["topic_mastery"]:
                    result += f"- {topic['topic']}: {topic['mastery']:.1f}% ({topic['attempts']} attempts)\n"
            else:
                result += "- Not enough data to assess topic mastery yet.\n"

            # Learning Preferences
            prefs = report["learning_preferences"]
            result += "\n## Learning Preferences\n"
            result += f"- Preferred Learning Style: {prefs['preferred_style'] or 'Not enough data'}\n"
            result += f"- Optimal Difficulty Level: {prefs['optimal_difficulty'] or 'Not enough data'}\n"

            # Insights
            insights = report["insights"]
            result += "\n## Insights and Recommendations\n"
            if insights["recommendations"]:
                for rec in insights["recommendations"]:
                    result += f"- {rec['suggestion']}\n"
            else:
                result += "- Continue learning to receive personalized recommendations.\n"

            return result
        except Exception as e:
            error_msg = f"Error generating progress report: {str(e)}"
            logger.error(error_msg)
            return f"Something went wrong while generating the progress report. Please try again later.\n\nTechnical details: {str(e)}"

    def end_session_fn():
        """End session with proper error handling"""
        try:
            if not companion:
                return "Error: Learning Companion not initialized properly. Check logs for details."

            result = companion.end_session()
            return result
        except Exception as e:
            error_msg = f"Error ending session: {str(e)}"
            logger.error(error_msg)
            return f"Something went wrong while ending the session. Your progress might not be saved.\n\nTechnical details: {str(e)}"

    def visualize_learning_fn():
        """Visualize learning with proper error handling"""
        try:
            if not companion:
                return None  # Return None for the image display

            companion.visualize_learning()
            return "visualizations/learning_curves.png"  # Return path to image
        except Exception as e:
            error_msg = f"Error visualizing learning curves: {str(e)}"
            logger.error(error_msg)
            return None  # Return None for the image display

    # Create Gradio interface with better component definitions
    with gr.Blocks(title="Adaptive Learning Companion") as interface:
        gr.Markdown("# 🧠 Adaptive Learning Companion")
        gr.Markdown("An AI-powered personalized learning system with automatic adaptation to your learning style and performance.")

        with gr.Tab("Learn"):
            with gr.Row():
                with gr.Column(scale=3):
                    topic_input = gr.Textbox(
                        label="Topic",
                        placeholder="Enter a topic to learn (e.g., 'Python functions', 'Photosynthesis', 'Linear equations')",
                        interactive=True
                    )
                    content_type = gr.Radio(
                        choices=["full", "lesson", "quiz"],
                        label="Content Type",
                        value="full",
                        interactive=True
                    )
                    generate_btn = gr.Button("Generate Learning Content", variant="primary")

                with gr.Column(scale=7):
                    content_output = gr.Markdown(label="Learning Content")

            with gr.Row():
                user_answer = gr.Textbox(
                    label="Your Answer",
                    placeholder="Type your answer here...",
                    interactive=True
                )
                evaluation = gr.Radio(
                    choices=["Correct", "Incorrect"],
                    label="Mark Answer As",
                    interactive=True
                )
                evaluate_btn = gr.Button("Submit and Evaluate", variant="primary")

            feedback_output = gr.Markdown(label="Feedback")

        with gr.Tab("Progress & Insights"):
            with gr.Row():
                insights_btn = gr.Button("Get Learning Insights", variant="primary")
                report_btn = gr.Button("Generate Full Progress Report", variant="primary")

            with gr.Row():
                insights_output = gr.Markdown(label="Learning Insights")

            with gr.Row():
                report_output = gr.Markdown(label="Progress Report")

            with gr.Row():
                visualize_btn = gr.Button("Visualize Learning", variant="primary")
                visualization_output = gr.Image(label="Learning Curves")

        with gr.Tab("Session Management"):
            with gr.Row():
                end_session_btn = gr.Button("End Current Session", variant="primary")
                session_output = gr.Textbox(label="Session Status")

        # Set up event handlers with explicit parameters
        generate_btn.click(
            fn=generate_content_fn,
            inputs=[topic_input, content_type],
            outputs=content_output,
            api_name="generate_content"
        )

        evaluate_btn.click(
            fn=evaluate_answer_fn,
            inputs=[user_answer, evaluation],
            outputs=feedback_output,
            api_name="evaluate_answer"
        )

        insights_btn.click(
            fn=get_insights_fn,
            inputs=None,
            outputs=insights_output,
            api_name="get_insights"
        )

        report_btn.click(
            fn=get_progress_report_fn,
            inputs=None,
            outputs=report_output,
            api_name="get_progress_report"
        )

        visualize_btn.click(
            fn=visualize_learning_fn,
            inputs=None,
            outputs=visualization_output,
            api_name="visualize_learning"
        )

        end_session_btn.click(
            fn=end_session_fn,
            inputs=None,
            outputs=session_output,
            api_name="end_session"
        )

    return interface

# =============== 6. APPLICATION ENTRY POINT ===============

if __name__ == "__main__":
    # Create directory for user profiles if it doesn't exist
    os.makedirs("user_profiles", exist_ok=True)

    # Ensure directory for saving visualizations exists
    os.makedirs("visualizations", exist_ok=True)

    try:
        # Create and launch interface with appropriate settings
        interface = create_interface()

        # Launch with more robust configuration
        interface.launch(server_name="0.0.0.0")
        logger.info("Learning Companion launched successfully")
    except Exception as e:
        logger.critical(f"Failed to launch Learning Companion interface: {str(e)}")
