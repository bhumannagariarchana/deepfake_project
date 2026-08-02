class ChallengeVerifier:
    def __init__(self, challenge_sequence, consecutive_required=5, timeout_seconds=20.0):
        self.challenge_sequence = challenge_sequence
        self.current_idx = 0
        self.consecutive_required = consecutive_required
        
        # State variables
        self.consecutive_count = 0
        self.completed_challenges = []
        self.is_completed = False
        self.failed = False
        
        # Blink counting state
        self.blink_count = 0
        self.eye_closed = False

    def reset_action_state(self):
        self.consecutive_count = 0
        self.blink_count = 0
        self.eye_closed = False

    def verify_metric(self, action, metrics, baselines):
        """
        Evaluates if the metric condition for 'action' is met.
        Returns (is_performing: bool, progress: float)
        """
        if action == "Smile":
            # Mouth width / eye distance increases relative to baseline
            ratio = metrics["smile"] / baselines["smile"] if baselines["smile"] > 0 else 1.0
            is_performing = ratio >= 1.15
            progress = min(1.0, max(0.0, (ratio - 1.0) / 0.15))
            return is_performing, progress

        elif action == "Blink":
            # Eye closes: EAR drops below 55% of baseline
            ear_ratio = metrics["ear"] / baselines["ear"] if baselines["ear"] > 0 else 1.0
            is_closed = ear_ratio <= 0.55
            progress = 1.0 if is_closed else 0.0
            return is_closed, progress

        elif action == "Blink twice":
            ear_ratio = metrics["ear"] / baselines["ear"] if baselines["ear"] > 0 else 1.0
            is_closed = ear_ratio <= 0.55
            progress = 1.0 if is_closed else 0.0
            return is_closed, progress

        elif action == "Turn Left":
            # Yaw ratio increases (nose moves right towards screen right / subject's left)
            is_performing = metrics["yaw"] >= 0.62
            progress = min(1.0, max(0.0, (metrics["yaw"] - 0.5) / 0.12))
            return is_performing, progress

        elif action == "Turn Right":
            # Yaw ratio decreases (nose moves left towards screen left / subject's right)
            is_performing = metrics["yaw"] <= 0.38
            progress = min(1.0, max(0.0, (0.5 - metrics["yaw"]) / 0.12))
            return is_performing, progress

        elif action == "Look Up":
            # Pitch ratio decreases (nose moves closer to forehead)
            ratio = metrics["pitch"] / baselines["pitch"] if baselines["pitch"] > 0 else 1.0
            is_performing = ratio <= 0.82
            progress = min(1.0, max(0.0, (1.0 - ratio) / 0.18))
            return is_performing, progress

        elif action == "Look Down":
            # Pitch ratio increases (nose moves closer to chin)
            ratio = metrics["pitch"] / baselines["pitch"] if baselines["pitch"] > 0 else 1.0
            is_performing = ratio >= 1.22
            progress = min(1.0, max(0.0, (ratio - 1.0) / 0.22))
            return is_performing, progress

        elif action == "Raise Eyebrows":
            # Eyebrow distance increases relative to baseline
            ratio = metrics["eyebrow"] / baselines["eyebrow"] if baselines["eyebrow"] > 0 else 1.0
            is_performing = ratio >= 1.12
            progress = min(1.0, max(0.0, (ratio - 1.0) / 0.12))
            return is_performing, progress

        elif action == "Open Mouth":
            # MAR is high
            is_performing = metrics["mar"] >= 0.14
            progress = min(1.0, max(0.0, metrics["mar"] / 0.14))
            return is_performing, progress

        elif action == "Close Mouth":
            # MAR is low
            is_performing = metrics["mar"] <= 0.04
            progress = min(1.0, max(0.0, (0.1 - metrics["mar"]) / 0.06))
            return is_performing, progress

        return False, 0.0

    def update(self, metrics, baselines):
        """
        Updates the verifier state with new frame metrics.
        Returns:
            dict: Current challenge verification status.
        """
        if self.is_completed or self.failed:
            return self.get_status()

        if self.current_idx >= len(self.challenge_sequence):
            self.is_completed = True
            return self.get_status()

        current_action = self.challenge_sequence[self.current_idx]
        is_performing, progress = self.verify_metric(current_action, metrics, baselines)

        # Handle Blink / Blink twice differently using a state machine
        if current_action in ["Blink", "Blink twice"]:
            target_blinks = 2 if current_action == "Blink twice" else 1
            ear_ratio = metrics["ear"] / baselines["ear"] if baselines["ear"] > 0 else 1.0
            
            # Blink logic: transitions from OPEN -> CLOSED -> OPEN
            if ear_ratio <= 0.55:
                # Eyelid is closed
                self.eye_closed = True
            else:
                # Eyelid is open
                if self.eye_closed:
                    # Registrater a complete blink on transition from CLOSED -> OPEN
                    self.blink_count += 1
                    self.eye_closed = False

            # Progress is how many blinks completed relative to target
            progress = min(1.0, self.blink_count / target_blinks)

            if self.blink_count >= target_blinks:
                self.completed_challenges.append(current_action)
                self.current_idx += 1
                self.reset_action_state()
        else:
            # For continuous gestures (Smile, Turn, Eyebrows, Open Mouth)
            if is_performing:
                self.consecutive_count += 1
                if self.consecutive_count >= self.consecutive_required:
                    # Action complete!
                    self.completed_challenges.append(current_action)
                    self.current_idx += 1
                    self.reset_action_state()
            else:
                # Reset consecutive frames if they stop performing the action before completion
                self.consecutive_count = max(0, self.consecutive_count - 1)

        # Re-check if we reached the end after updating
        if self.current_idx >= len(self.challenge_sequence):
            self.is_completed = True

        return self.get_status()

    def get_status(self):
        current_action = (
            self.challenge_sequence[self.current_idx]
            if self.current_idx < len(self.challenge_sequence)
            else None
        )
        
        # Calculate active progress percentage
        if current_action in ["Blink", "Blink twice"]:
            active_progress = self.blink_count / (2 if current_action == "Blink twice" else 1)
        else:
            active_progress = self.consecutive_count / self.consecutive_required if current_action else 0.0

        return {
            "challenge_sequence": self.challenge_sequence,
            "current_idx": self.current_idx,
            "current_action": current_action,
            "active_progress": active_progress,
            "completed_challenges": self.completed_challenges,
            "is_completed": self.is_completed,
            "failed": self.failed
        }
