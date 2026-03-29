"""Task bank: diverse agent task specifications for evaluation.

Each task defines the tool/file profile that drives realistic trace
generation.  Covers simple → complex tasks across 7 domains so the
evaluation tests cross-domain generalization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class TaskSpec:
    """Specification for a single agent coding task."""

    name: str
    description: str
    prompt: str
    domain: str  # web_app, cli_tool, data_pipeline, game, docs, testing, devops
    complexity: str  # simple, medium, complex
    typical_tools: list[str] = field(default_factory=list)
    typical_files: list[str] = field(default_factory=list)
    estimated_steps: int = 300
    # Phase profile: relative proportion of steps in each phase
    phase_profile: dict[str, float] = field(default_factory=dict)


# ── Task definitions ──────────────────────────────────────────────────────

TASK_BANK: list[TaskSpec] = [
    # ── Simple tasks ──────────────────────────────────────────────────
    TaskSpec(
        name="rest_api",
        description="Build a REST API with CRUD endpoints",
        prompt="Create a Flask REST API with user CRUD endpoints and SQLite storage",
        domain="web_app",
        complexity="simple",
        typical_tools=["read", "write", "edit", "bash", "grep", "glob"],
        typical_files=[
            "app.py", "models.py", "routes.py", "requirements.txt",
            "tests/test_api.py", "config.py", "README.md",
        ],
        estimated_steps=250,
        phase_profile={"planning": 0.10, "executing": 0.60, "verifying": 0.20, "iterating": 0.10},
    ),
    TaskSpec(
        name="markdown_docs",
        description="Generate a documentation site from source code",
        prompt="Create markdown documentation for this Python package with API reference",
        domain="docs",
        complexity="simple",
        typical_tools=["read", "write", "grep", "glob", "bash"],
        typical_files=[
            "docs/index.md", "docs/api.md", "docs/getting_started.md",
            "docs/conf.py", "mkdocs.yml", "src/__init__.py", "src/core.py",
        ],
        estimated_steps=200,
        phase_profile={"planning": 0.15, "executing": 0.55, "verifying": 0.15, "iterating": 0.15},
    ),
    TaskSpec(
        name="bash_utility",
        description="Write a bash deployment script with rollback",
        prompt="Create a bash script that deploys a Docker container with health checks and rollback",
        domain="devops",
        complexity="simple",
        typical_tools=["read", "write", "bash", "edit", "grep"],
        typical_files=[
            "deploy.sh", "rollback.sh", "Dockerfile", "docker-compose.yml",
            ".env.example", "health_check.sh", "tests/test_deploy.sh",
        ],
        estimated_steps=180,
        phase_profile={"planning": 0.15, "executing": 0.50, "verifying": 0.25, "iterating": 0.10},
    ),
    TaskSpec(
        name="unit_test_suite",
        description="Write comprehensive unit tests for an existing module",
        prompt="Write pytest tests for the authentication module with mocking and fixtures",
        domain="testing",
        complexity="simple",
        typical_tools=["read", "write", "bash", "grep", "glob"],
        typical_files=[
            "src/auth.py", "src/models.py", "tests/test_auth.py",
            "tests/conftest.py", "tests/fixtures.py", "pytest.ini",
        ],
        estimated_steps=220,
        phase_profile={"planning": 0.10, "executing": 0.50, "verifying": 0.30, "iterating": 0.10},
    ),

    # ── Medium tasks ──────────────────────────────────────────────────
    TaskSpec(
        name="react_dashboard",
        description="Build a React analytics dashboard with charts",
        prompt="Create a React dashboard with D3 charts showing user analytics data",
        domain="web_app",
        complexity="medium",
        typical_tools=["read", "write", "edit", "bash", "grep", "glob"],
        typical_files=[
            "src/App.tsx", "src/components/Chart.tsx", "src/components/Dashboard.tsx",
            "src/components/Sidebar.tsx", "src/hooks/useData.ts", "src/api/client.ts",
            "src/types.ts", "package.json", "tsconfig.json", "vite.config.ts",
            "tests/Dashboard.test.tsx", "src/styles/dashboard.css",
        ],
        estimated_steps=350,
        phase_profile={"planning": 0.12, "executing": 0.55, "verifying": 0.18, "iterating": 0.15},
    ),
    TaskSpec(
        name="cli_data_processor",
        description="Build a CLI data processing pipeline",
        prompt="Create a Click CLI tool that reads CSV/JSON, transforms data, and outputs reports",
        domain="cli_tool",
        complexity="medium",
        typical_tools=["read", "write", "edit", "bash", "grep", "glob"],
        typical_files=[
            "cli.py", "processor.py", "transformers.py", "validators.py",
            "output.py", "tests/test_cli.py", "tests/test_processor.py",
            "setup.py", "README.md", "sample_data/input.csv",
        ],
        estimated_steps=280,
        phase_profile={"planning": 0.12, "executing": 0.55, "verifying": 0.20, "iterating": 0.13},
    ),
    TaskSpec(
        name="chat_app",
        description="Build a WebSocket chat application",
        prompt="Create a real-time chat app with FastAPI WebSockets and a simple HTML frontend",
        domain="web_app",
        complexity="medium",
        typical_tools=["read", "write", "edit", "bash", "grep", "glob"],
        typical_files=[
            "server.py", "models.py", "auth.py", "static/index.html",
            "static/chat.js", "static/style.css", "tests/test_server.py",
            "tests/test_websocket.py", "requirements.txt", "Dockerfile",
        ],
        estimated_steps=300,
        phase_profile={"planning": 0.12, "executing": 0.55, "verifying": 0.20, "iterating": 0.13},
    ),
    TaskSpec(
        name="etl_pipeline",
        description="Build an ETL data pipeline with validation",
        prompt="Create a data pipeline that extracts from PostgreSQL, transforms with pandas, loads to S3",
        domain="data_pipeline",
        complexity="medium",
        typical_tools=["read", "write", "edit", "bash", "grep", "glob"],
        typical_files=[
            "extract.py", "transform.py", "load.py", "pipeline.py",
            "validators.py", "config.yaml", "tests/test_extract.py",
            "tests/test_transform.py", "tests/test_pipeline.py",
            "requirements.txt", "Makefile",
        ],
        estimated_steps=320,
        phase_profile={"planning": 0.15, "executing": 0.50, "verifying": 0.20, "iterating": 0.15},
    ),

    # ── Complex tasks ─────────────────────────────────────────────────
    TaskSpec(
        name="fullstack_auth",
        description="Full-stack app with authentication and authorization",
        prompt="Build a FastAPI + React app with JWT auth, role-based access, and user management",
        domain="web_app",
        complexity="complex",
        typical_tools=["read", "write", "edit", "bash", "grep", "glob"],
        typical_files=[
            "backend/main.py", "backend/auth.py", "backend/models.py",
            "backend/routes/users.py", "backend/routes/admin.py",
            "backend/middleware.py", "backend/database.py",
            "frontend/src/App.tsx", "frontend/src/pages/Login.tsx",
            "frontend/src/pages/Dashboard.tsx", "frontend/src/api/auth.ts",
            "frontend/src/context/AuthContext.tsx",
            "tests/test_auth.py", "tests/test_routes.py",
            "docker-compose.yml", "alembic.ini",
        ],
        estimated_steps=450,
        phase_profile={"planning": 0.15, "executing": 0.50, "verifying": 0.20, "iterating": 0.15},
    ),
    TaskSpec(
        name="game_physics",
        description="2D game with physics engine",
        prompt="Create a 2D platformer game with Pygame including collision detection and particle effects",
        domain="game",
        complexity="complex",
        typical_tools=["read", "write", "edit", "bash", "grep", "glob"],
        typical_files=[
            "main.py", "engine/physics.py", "engine/renderer.py",
            "engine/collision.py", "engine/particles.py",
            "entities/player.py", "entities/platform.py", "entities/enemy.py",
            "levels/level1.json", "assets/config.json",
            "tests/test_physics.py", "tests/test_collision.py",
            "requirements.txt", "README.md",
        ],
        estimated_steps=400,
        phase_profile={"planning": 0.15, "executing": 0.55, "verifying": 0.15, "iterating": 0.15},
    ),
    TaskSpec(
        name="ml_pipeline",
        description="ML training pipeline with experiment tracking",
        prompt="Build a scikit-learn pipeline with hyperparameter tuning, MLflow tracking, and model serving",
        domain="data_pipeline",
        complexity="complex",
        typical_tools=["read", "write", "edit", "bash", "grep", "glob"],
        typical_files=[
            "train.py", "evaluate.py", "serve.py", "pipeline.py",
            "features.py", "config/experiment.yaml", "config/model.yaml",
            "data/preprocess.py", "tests/test_pipeline.py",
            "tests/test_features.py", "notebooks/exploration.ipynb",
            "Makefile", "requirements.txt", "Dockerfile",
        ],
        estimated_steps=380,
        phase_profile={"planning": 0.15, "executing": 0.50, "verifying": 0.20, "iterating": 0.15},
    ),
    TaskSpec(
        name="ci_cd_setup",
        description="CI/CD pipeline with GitHub Actions",
        prompt="Set up GitHub Actions CI/CD with linting, testing, Docker build, and deploy to AWS ECS",
        domain="devops",
        complexity="complex",
        typical_tools=["read", "write", "edit", "bash", "grep", "glob"],
        typical_files=[
            ".github/workflows/ci.yml", ".github/workflows/deploy.yml",
            "Dockerfile", "docker-compose.yml", "Makefile",
            "scripts/deploy.sh", "scripts/health_check.sh",
            "terraform/main.tf", "terraform/ecs.tf", "terraform/variables.tf",
            ".env.example", "tests/test_deploy.py",
        ],
        estimated_steps=350,
        phase_profile={"planning": 0.20, "executing": 0.45, "verifying": 0.20, "iterating": 0.15},
    ),
    TaskSpec(
        name="api_test_harness",
        description="API integration test harness with load testing",
        prompt="Build a comprehensive API test harness using pytest and locust for load testing",
        domain="testing",
        complexity="medium",
        typical_tools=["read", "write", "edit", "bash", "grep", "glob"],
        typical_files=[
            "tests/conftest.py", "tests/test_endpoints.py",
            "tests/test_auth_flow.py", "tests/test_edge_cases.py",
            "load_tests/locustfile.py", "load_tests/config.py",
            "fixtures/users.json", "fixtures/data.json",
            "helpers/api_client.py", "helpers/assertions.py",
            "pytest.ini", "requirements-test.txt",
        ],
        estimated_steps=300,
        phase_profile={"planning": 0.12, "executing": 0.50, "verifying": 0.25, "iterating": 0.13},
    ),
]


def get_task(name: str) -> TaskSpec:
    """Look up a task by name."""
    for task in TASK_BANK:
        if task.name == name:
            return task
    raise KeyError(f"Unknown task: {name}")


def get_tasks_by_domain(domain: str) -> list[TaskSpec]:
    """Get all tasks in a domain."""
    return [t for t in TASK_BANK if t.domain == domain]


def get_tasks_by_complexity(complexity: str) -> list[TaskSpec]:
    """Get all tasks of a given complexity."""
    return [t for t in TASK_BANK if t.complexity == complexity]
