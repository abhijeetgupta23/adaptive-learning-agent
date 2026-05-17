# MDA: A Formal Approach to Game Design and Game Research

Citation: Hunicke, R., LeBlanc, M., & Zubek, R. (2004). *MDA: A Formal Approach to Game Design and Game Research.* AAAI Workshop on Challenges in Game AI.

MDA decomposes any game into three layered components: **Mechanics** are the data and algorithms the designer ships — the rules, the moves, the discrete actions a player can take. **Dynamics** are the run-time behavior that emerges when those mechanics meet a real player making real choices — pacing, tension, recovery loops, the "what actually happens at the table." **Aesthetics** are the emotional responses the dynamics evoke — fellowship, challenge, discovery, fantasy, expression, narrative, sensation, submission (the eight aesthetic categories the paper enumerates).

The framework's central claim is that **designer and player approach the game from opposite ends**. The designer writes mechanics and *hopes* certain dynamics emerge, which *should* produce intended aesthetics. The player feels the aesthetics first, infers the dynamics from play, and only later (if at all) reasons about the mechanics. Misalignment between the two ends is the dominant failure mode of designed games — including educational games where the designer authors a "learning mechanic" but never tests what aesthetic the player actually feels.

For a generation system, MDA gives three concrete leverage points. First, the designer agent should commit to an intended aesthetic *before* writing any mechanics — otherwise mechanics get authored by reflex toward whatever genre the agent has seen most. Second, dynamics are the agent's blind spot: mechanics and aesthetics can both be inspected statically, but dynamics require simulation. A good agent compensates by predicting dynamics explicitly (a "core loop" commitment) rather than letting them emerge unchecked. Third, the aesthetic vocabulary (mastery, aha, curiosity, flow, surprise) is a small, structured search space — far more tractable than freeform genre choice. Constraining the agent's aesthetic commitment to that vocabulary is a high-leverage prompt move.

In a Game-Based Learning context, MDA reframes "is this game educational?" as "do its dynamics reliably produce the aesthetic of *understanding*?" — which is what the learning-link/core-loop/aesthetic commit sequence operationalizes.

Source: https://users.cs.northwestern.edu/~hunicke/MDA.pdf
