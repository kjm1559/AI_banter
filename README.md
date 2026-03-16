# AI Banter - Multi-AI Podcast Script Generator

## Overview

AI Banter is an autonomous agent that generates 1-hour podcast-style conversation scripts featuring multiple AI personalities. The system coordinates dynamic dialogues between a moderator and expert panelists, each with distinct personas, to create engaging and coherent discussions on any given topic.

## Features

### Core Capabilities

1. **Multi-AI Dialogue Generation**
   - Creates 1-hour conversation scripts between multiple AI speakers
   - Supports flexible participant counts (minimum 3 including moderator)

2. **Dynamic Persona Assignment**
   - Automatically assigns expert personas to panelists based on the topic
   - Each panelist has a unique perspective and expertise area
   - Moderator controls conversation flow and pacing

3. **Topic-Driven Architecture**
   - Accepts custom topics as input
   - Generates relevant expert personas tailored to the subject
   - Ensures coherent discussion progression

4. **Role-Based AI Orchestration**
   - Rotates speakers dynamically based on conversation context
   - Each AI receives role-specific system prompts
   - Chat-style message format for natural dialogue

5. **Conversational Flow Management**
   - Moderator AI guides discussion toward engaging directions
   - Maintains logical transitions between topics
   - Builds natural conversation momentum

6. **Multi-Language Support**
   - Configurable output language via input
   - Default language: English

7. **Podcast-Style Format**
   - Structured for audio production
   - Natural spoken language patterns
   - Includes conversational elements (reactions, follow-ups, clarifications)

8. **Iterative Script Building**
   - Generates conversation flow guide at startup
   - Sequential speaker invocation with context accumulation
   - Real-time script updates with dialogue history
   - Progressively builds complete 1-hour script

## System Architecture

```
┌─────────────────────────────────────────┐
│         Orchestration Agent             │
│  - Conversation Flow Guide Generator    │  
│  - Speaker Turn Management              │
│  - Script Assembly & Updates            │
└─────────────────┬───────────────────────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
┌──────────────┐    ┌──────────────┐
│  Moderator   │    │   Panelist   │
│   (Host)     │◄───┤  (Expert 1)  │
└──────────────┘    └──────────────┘
                      │         │
                      ▼         ▼
               ┌──────────────┐ ┌──────────────┐
               │   Panelist   │ │   Panelist   │
               │  (Expert 2)  │ │  (Expert 3+) │
               └──────────────┘ └──────────────┘
```

## Agent Workflow

1. **Initialization**
   - Receive inputs: number of participants, topic, language
   - Validate minimum 3 participants (including moderator)

2. **Persona Creation**
   - Generate expert personas for panelists based on topic
   - Assign moderator role for flow control

3. **Flow Guide Generation**
   - Create conversation structure and key discussion points
   - Outline segment breakdown for 1-hour duration

4. **Dialogue Execution**
   - Select next speaker based on context
   - Invoke AI with:
     - Role-specific system prompt
     - Full conversation history
     - Current topic/sub-topic context
   - Receive generated response

5. **Script Assembly**
   - Append new dialogue to script
   - Update conversation state
   - Track progress toward 1-hour target

6. **Iteration**
   - Repeat steps 4-5 until complete
   - Ensure coherent transitions throughout

7. **Final Output**
   - Complete 1-hour podcast script
   - All dialogues in format: `[Speaker Name]: Dialogue`

## Input Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `num_participants` | Integer | Yes | - | Number of speakers (min: 3, includes moderator) |
| `topic` | String | Yes | - | Discussion topic/subject |
| `language` | String | No | "english" | Output language for dialogue |

## Output Format

```
[MODERATOR - Name]: Welcome to our discussion on {topic}. Let's begin...

[EXPERT 1 - Category Specialist]: That's a fascinating topic. From my perspective...

[EXPERT 2 - Technology Analyst]: I'd like to add to that point...

[MODERATOR - Name]: Excellent insights. Let's dive deeper into...

... [continues for 1-hour duration]
```

## Technical Requirements

- LLM integration for AI role-playing
- Context window management for conversation history
- Turn-taking algorithm for natural flow
- Script tracking and progress measurement

## Usage Example

```yaml
num_participants: 5
topic: "The Future of Artificial Intelligence in Healthcare"
language: "english"
```

**Expected Output:**
- Moderator + 4 expert panelists
- 1-hour script (~9000-10000 words)
- Natural podcast-style conversation
- Coherent topic progression

## Development Guidelines

See [AGENTS.md](./AGENTS.md) for project code of conduct and development principles.

## Future Enhancements

- Variable duration support (not fixed to 1 hour)
- Multiple discussion rounds/topics in single session
- Audience Q&A generation
- Export formats (PDF, TXT, Markdown, audio transcript)
- Topic-specific persona templates
- Conversation complexity adjustment

---

*Built for creating engaging, multi-perspective AI dialogues in podcast format.*
