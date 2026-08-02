import gradio as gr
import spaces
from gradio.routes import App
from src.routers.router import router as ai_router

@spaces.GPU
def status_checker(query_text: str = ""):
    if not query_text:
        return "AI Service Online! REST API /ai/search dan /ai/summarize aktif."
    return f"AI Service Online & Siap! Query dites: '{query_text}'"

demo = gr.Interface(
    fn=status_checker,
    inputs=gr.Textbox(lines=2, placeholder="Tes query di sini..."),
    outputs="text",
    title="🐠 AI Service Skripsi API",
    description="REST API AI Service aktif dan melayani endpoint /ai/search, /ai/summarize, dan /health"
)

# ──────────────────────────────────────────────────────────
# PATCH: intercept App.create_app so that whenever demo.launch()
# (or spaces wrapper) calls it, we get a chance to add our REST routes
# ──────────────────────────────────────────────────────────
_original_create_app = App.create_app  # already a plain function in Python 3.10

@staticmethod
def _patched_create_app(blocks, **kwargs):
    # Remove 'app' kwarg if present since HF Space Gradio 4.44.0 doesn't support it
    kwargs.pop("app", None)
    gradio_app = _original_create_app(blocks, **kwargs)
    # Register REST API routes on the freshly created Gradio App
    gradio_app.include_router(ai_router, prefix="/ai")
    gradio_app.add_api_route("/health", lambda: {"status": "ok", "service": "AI-Service-Skripsi"}, methods=["GET"])
    return gradio_app

App.create_app = _patched_create_app
# ──────────────────────────────────────────────────────────

demo.launch(server_name="0.0.0.0", server_port=7860)
