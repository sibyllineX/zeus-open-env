from __future__ import annotations

try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    # Fallback: create a minimal FastAPI app manually
    from fastapi import FastAPI, Request

    def create_app(env_cls, action_cls, observation_cls, env_name="leakhunter"):
        app = FastAPI(title=env_name)
        env = env_cls()

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        @app.get("/state")
        async def get_state():
            return env.state.model_dump()

        @app.post("/reset")
        async def reset(request: Request):
            body = await request.json()
            obs = env.reset(**body)
            return {"observation": obs.model_dump(), "done": obs.done, "reward": obs.reward}

        @app.post("/step")
        async def step(request: Request):
            body = await request.json()
            action = action_cls.model_validate(body["action"])
            obs = env.step(action)
            return {"observation": obs.model_dump(), "done": obs.done, "reward": obs.reward}

        return app

try:
    from ..models import LeakHunterAction, LeakHunterObservation
    from .environment import LeakHunterEnvironment
except ImportError:
    from models import LeakHunterAction, LeakHunterObservation
    from server.environment import LeakHunterEnvironment

app = create_app(
    LeakHunterEnvironment,
    LeakHunterAction,
    LeakHunterObservation,
    env_name="leakhunter",
)

def main():
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
