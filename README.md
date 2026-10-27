# Verifiable Robot Agent Harness

An extensible agent runtime for verifiable robot task execution in simulation.
The project combines planner/tool/runtime orchestration, failure-aware recovery,
structured skill policies, action-chunk execution boundaries, and test-time
action selection behind a shared harness and verifier.

## System Overview

```text
Task / observation
        |
        v
Planner -> Toolkit schema validation -> Harness runtime -> Simulator
                                                       |
                                                       v
                                      Verifier / commit gate / failure trace

Policy backends share the same execution boundary:
  - Transformer skill policy
  - π0.7-compatible action-chunk interface
  - JitRL-inspired test-time selector
```

The runtime is deliberately split into decision, execution, and verification
layers. A policy can propose a skill or an action chunk, but only registered
toolkit operations and harness preconditions can reach the simulator.

## Implemented Components

- Planner loop with typed toolkit calls and tool-result feedback.
- Harness precondition checks and structured execution traces.
- Desktop pick-and-place simulator with scene state and path constraints.
- Verifier gates for grasp, placement, and final task state.
- Transformer policy that fuses scene, task, and robot-state tokens into a
  structured skill call.
- π0.7-compatible adapter boundary for continuous action chunks, with a local
  deterministic fallback for interface testing.
- JitRL-inspired discrete selector using retrieved transitions and advantage
  adjusted candidate scores without changing model weights.
- Parallel executor routing so policy backends share the same simulator,
  verifier, and safety boundary.

The π0.7 adapter is an integration boundary and local fallback. It does not
claim to bundle or deploy an upstream π0.7 checkpoint.

## Requirements

- Python 3.12+
- PyTorch with a CUDA build is recommended for policy experiments.
- The simulator and harness can run without a physical robot.

Install the Python dependencies in a virtual environment. For CUDA-enabled
PyTorch, use the wheel index appropriate for the target GPU and CUDA runtime.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the Demos

From the repository root:

```bash
PYTHONPATH=. python scripts/run_lesson24.py
PYTHONPATH=. python scripts/run_lesson25.py
PYTHONPATH=. python scripts/run_jitrl_demo.py
```

The first command runs the planner, toolkit, harness, simulator, and verifier
loop. The second prints the Transformer attention and structured skill-call
experiment. The third runs the JitRL-inspired test-time selector.

## Configuration

Copy `.env.example` to `.env` and provide credentials only for the model
provider you explicitly use. `.env` is ignored by Git. The offline simulator
and local policy demos do not require an API key.

## Project Layout

```text
robot_agent_harness/
  agent_client/   model-provider boundary
  controllers/    tensor and Transformer controllers
  memory/         episodic transitions and JitRL-inspired selection
  planner/        planner/tool/runtime loop
  policies/       action-chunk and executor adapters
  runtime/        harness dispatch and preconditions
  simulator/      deterministic simulated world
  toolkit/        registered skill schemas
  verifier/       commit gates and task verification
scripts/          runnable integration demos
configs/          scene and skill configuration
```

## Scope and Attribution

The architecture is a clean-room implementation inspired by public ideas from
agent runtimes, verifiable web/robot environments, VLA action chunks, and
test-time reinforcement learning. It does not include upstream source code or
claim reproduction of upstream benchmark numbers.

## License

This repository currently has no declared open-source license. Add one before
accepting external contributions or redistributing the code.
