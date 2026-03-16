"""Main agent module that orchestrates AI Banter script generation."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from src.config import Config, load_config
from src.flow_generator import FlowGuideGenerator
from src.orchestrator import ConversationOrchestrator
from src.persona_manager import PersonaManager
from src.types import ConversationState, Persona, Script


class AIBanterAgent:
    """Main orchestration agent for AI Banter podcast script generation.
    
    Coordinates all components to generate multi-AI dialogue scripts.
    """

    def __init__(self, config: Config):
        """Initialize the agent with all required components.

        Args:
            config: Configuration object with OpenAI credentials.

        """
        self.config = config
        self.persona_manager = PersonaManager(config)
        self.flow_generator = FlowGuideGenerator(config)
        self.orchestrator = ConversationOrchestrator(config)

    def generate_script(
        self,
        topic: str,
        num_participants: int,
        language: str = "english",
        duration_mins: int = 60
    ) -> Script:
        """
        Main entry point for script generation.
        
        Workflow:
        1. Validate inputs (min 3 participants)
        2. Generate personas (moderator + experts)
        3. Generate flow guide for conversation
        4. Initialize conversation state
        5. Loop:
           a. Check if conversation complete
           b. Select next speaker
           c. Build context prompt
           d. Invoke AI for response
           e. Add turn to script
           f. Update state
           g. Check progress
        6. Return assembled script
        
        Args:
            topic: Discussion topic
            num_participants: Number of speakers (min: 3)
            language: Output language (default: english)
            duration_mins: Target duration (default: 60)
        
        Returns:
            Complete Script object
            
        Raises:
            ValueError: If input validation fails
            RuntimeError: If generation fails
        """
        print("=" * 60)
        print("AI BANTER - Podcast Script Generator")
        print("=" * 60)
        print(f"\nTopic: {topic}")
        print(f"Participants: {num_participants}")
        print(f"Language: {language}")
        print(f"Duration: {duration_mins} minutes")
        print("-" * 60)

        # Step 1: Validate inputs
        print("\n[1/6] Validating inputs...")
        self._validate_inputs(num_participants)
        print("✓ Validation complete")

        # Step 2: Generate personas
        print("\n[2/6] Generating personas...")
        personas = self.persona_manager.generate_personas(topic, num_participants, language)
        print(f"✓ Generated {len(personas)} personas:")
        for persona in personas:
            print(f"    - {persona.name} ({persona.role}): {persona.expertise}")

        # Step 3: Generate flow guide
        print("\n[3/6] Generating conversation flow guide...")
        flow_guide = self.flow_generator.generate_flow_guide(topic, duration_mins)
        print(f"✓ Flow guide generated with {len(flow_guide.segments)} segments")
        print(f"  Total planned duration: {flow_guide.total_duration_mins} minutes")

        # Step 4: Initialize state
        print("\n[4/6] Initializing conversation...")
        state = ConversationState()
        turns = []
        print("✓ Conversation initialized")

        # Step 5: Generate conversation turns
        print("\n[5/6] Generating dialogue...")
        turn_num = 0
        target_words = duration_mins * 150  # ~150 words per minute

        while turn_num < 500:  # Safety limit
            # Check if conversation complete
            if self.flow_generator.is_conversation_complete(flow_guide, state.current_segment_idx):
                print("✓ All segments completed")
                break

            current_segment = self.flow_generator.get_current_segment(
                flow_guide, state.current_segment_idx
            )
            if current_segment is None:
                print("✓ Generation complete")
                break

            # Select next speaker
            next_speaker = self.orchestrator.select_next_speaker(state, personas)

            # Track progress every 10 turns
            if turn_num % 10 == 0:
                current_words = sum(t.word_count for t in turns)
                progress = min(100 * current_words / target_words, 100)
                print(f"  Turn {turn_num + 1}/{len(turns) + 1}: Progress ~{progress:.1f}%")

            # Build context and invoke speaker
            script_so_far = Script(
                turns=turns,
                topic=topic,
                language=language,
                total_word_count=sum(t.word_count for t in turns)
            )

            messages = self.orchestrator.build_context_prompt(
                persona=next_speaker,
                script=script_so_far,
                current_segment=current_segment,
                state=state,
                language=language
            )

            turn = self.orchestrator.invoke_speaker(
                persona=next_speaker,
                messages=messages,
                turn_number=turn_num + 1
            )

            # Add turn and update state
            turns.append(turn)
            state = self.orchestrator.update_state(state, next_speaker, turn)

            # Check if we should advance to next segment
            segment = flow_guide.segments[state.current_segment_idx]
            words_per_segment = target_words / len(flow_guide.segments)
            segment_word_count = sum(
                t.word_count for t in turns 
                if t.turn_number > turn_num - len(turns) + state.current_turn_count
            )

            # Advance segment if current segment has enough content
            if state.current_turn_count > 5:  # At least 5 turns per segment
                state.current_segment_idx += 1
                state.current_turn_count = 0

            turn_num += 1

            # Check word target completion
            current_script = Script(
                turns=turns,
                topic=topic,
                language=language,
                total_word_count=sum(t.word_count for t in turns)
            )
            completion = self.orchestrator.estimate_completion(current_script, target_words)
            if completion >= 1.0:
                print("✓ Target word count reached")
                break

        # Step 6: Assemble final script
        print("\n[6/6] Assembling final script...")
        total_word_count = sum(t.word_count for t in turns)
        script = Script(
            turns=turns,
            topic=topic,
            language=language,
            total_word_count=total_word_count
        )
        print(f"✓ Script assembled: {len(turns)} turns, {total_word_count} words")

        return script

    def _validate_inputs(self, num_participants: int) -> None:
        """Validate num_participants >= 3.

        Args:
            num_participants: Number of participants to validate.
            
        Raises:
            ValueError: If num_participants < 3.
        """
        if num_participants < 3:
            raise ValueError(
                f"num_participants must be at least 3 (got {num_participants}). "
                "Minimum includes 1 moderator + 2 experts."
            )

    def _format_output(self, script: Script) -> str:
        """Format script as readable podcast transcript.
        
        Args:
            script: The script to format.
            
        Returns:
            Formatted transcript string.
        """
        lines = [
            "=" * 60,
            f"PODCAST SCRIPT: {script.topic}",
            f"Language: {script.language}",
            f"Total Turns: {len(script.turns)}",
            f"Total Words: {script.total_word_count}",
            "=" * 60,
            ""
        ]

        for turn in script.turns:
            role_label = "MODERATOR" if turn.role == "moderator" else "EXPERT"
            lines.append(f"[{role_label} - {turn.speaker_name}]: {turn.content}")
            lines.append("")

        lines.append("=" * 60)
        lines.append("END OF SCRIPT")
        lines.append("=" * 60)

        return "\n".join(lines)

    def save_script(self, script: Script, filepath: str) -> None:
        """Save script to file.
        
        Args:
            script: The script to save.
            filepath: Path to save the script.
            
        Raises:
            IOError: If file cannot be written.
        """
        formatted = self._format_output(script)
        try:
            Path(filepath).write_text(formatted, encoding="utf-8")
            print(f"✓ Script saved to: {filepath}")
        except IOError as e:
            raise IOError(f"Failed to save script to {filepath}: {e}")


def main():
    """CLI entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="AI Banter - Multi-AI Podcast Script Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --topic "The Future of AI" --participants 4
  %(prog)s --topic "Climate Change" --participants 5 --language spanish --output script.txt
        """
    )

    parser.add_argument(
        "--topic", "-t",
        type=str,
        required=True,
        help="Discussion topic for the podcast"
    )

    parser.add_argument(
        "--participants", "-p",
        type=int,
        default=4,
        help="Number of speakers (min: 3, including moderator)"
    )

    parser.add_argument(
        "--language", "-l",
        type=str,
        default="english",
        help="Output language (default: english)"
    )

    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=60,
        help="Target duration in minutes (default: 60)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (if not specified, prints to stdout)"
    )

    args = parser.parse_args()

    try:
        # Load configuration
        print("Loading configuration...")
        config = load_config()
        print("✓ Configuration loaded")

        # Create agent
        print("Initializing AI Banter agent...")
        agent = AIBanterAgent(config)
        print("✓ Agent initialized")

        # Generate script
        print("\n")
        script = agent.generate_script(
            topic=args.topic,
            num_participants=args.participants,
            language=args.language,
            duration_mins=args.duration
        )

        # Format and output
        formatted = agent._format_output(script)

        if args.output:
            agent.save_script(script, args.output)
        else:
            print("\n" + formatted)

        print("\n✓ Generation complete!")
        sys.exit(0)

    except ValueError as e:
        print(f"\n❌ Validation Error: {e}", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"\n❌ IO Error: {e}", file=sys.stderr)
        sys.exit(2)
    except RuntimeError as e:
        print(f"\n❌ Runtime Error: {e}", file=sys.stderr)
        sys.exit(3)
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}", file=sys.stderr)
        sys.exit(99)


if __name__ == "__main__":
    main()
