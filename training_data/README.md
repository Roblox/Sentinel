# Training Data for Enhanced Sentinel Model

This directory contains comprehensive training data for detecting various types of content violations across multiple platforms.

## ⚠️ IMPORTANT USAGE RESTRICTIONS ⚠️

**This training data is intended EXCLUSIVELY for defensive security purposes:**

1. **Authorized Uses:**
   - Training content safety detection systems
   - Academic research on online toxicity
   - Developing content moderation tools
   - Safety system validation and testing

2. **PROHIBITED Uses:**
   - Generating harmful content
   - Amplifying toxic behaviors
   - Teaching harmful language patterns
   - Creating attack tools or systems

3. **Ethical Guidelines:**
   - Data should never be used to harm individuals or groups
   - All examples are for detection and prevention purposes only
   - Researchers must follow ethical AI development practices
   - Any use must comply with platform terms of service

## Data Structure

### Positive Examples (Violations)
- `child_safety_violations.json` - Child endangerment and grooming patterns
- `harassment_violations.json` - Harassment and cyberbullying examples  
- `hate_speech_violations.json` - Hate speech and discrimination content
- `gaming_toxicity_violations.json` - Gaming-specific toxic behavior
- `platform_abuse_violations.json` - Spam, scams, and platform manipulation
- `filter_bypass_violations.json` - Content filter circumvention attempts
- `coordinated_abuse_violations.json` - Organized harassment and brigading
- `violence_threats_violations.json` - Threats and violent content

### Negative Examples (Safe Content)
- `safe_gaming_content.json` - Legitimate gaming discussions
- `educational_content.json` - Learning and educational materials
- `general_conversation.json` - Normal social interactions
- `platform_help_content.json` - Support and assistance discussions
- `creative_content.json` - Art, writing, and creative expression

### Platform-Specific Data
- `discord_violations.json` - Discord-specific violation patterns
- `roblox_violations.json` - Roblox child safety and gaming violations
- `social_media_violations.json` - Twitter/Instagram/TikTok violations
- `forum_violations.json` - Reddit and forum-style violations

## Data Collection Methodology

All violation examples are:
1. **Synthetic or Anonymized**: No real user data is included
2. **Ethically Sourced**: Based on public research and documented patterns
3. **Diverse**: Covering multiple demographics, platforms, and violation types
4. **Validated**: Reviewed by safety experts for accuracy and appropriateness

## Usage Guidelines

When using this data:
1. Always maintain the defensive security context
2. Implement proper access controls and logging
3. Never expose raw violation examples to end users
4. Follow data protection and privacy regulations
5. Report any misuse or ethical concerns immediately

## Contributing

Contributions should:
- Follow ethical AI development practices
- Include only synthetic or properly anonymized examples
- Be reviewed by safety domain experts
- Maintain the defensive security focus
- Document sources and validation methods

For questions or concerns about this training data, please contact the security team.