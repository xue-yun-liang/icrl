# A multi-core processor design space exploration algorithm based on in-context RL

## Intro
The parsec-tests2 and m5out folders are used to put the sum of gem5&mcpat evaluation functions.

Since some of the previous reinforcement learning methods for processor design space exploration are one-time, that is, each time the constraint parameters are set, and then start training a new model suitable for the current scene, whenever the design requirements under the new scene (new constraints) are needed, the design needs to be re-designed from scratch, so the time cost is very high, this project is committed to applying In-Context RL to DSE, so as to realize the new constraint parameters that can modify the design framework and quickly apply it to the new scene.
