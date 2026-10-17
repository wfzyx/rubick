import argparse
import uvicorn

from rubick.config import settings


def main():
    parser = argparse.ArgumentParser(description="Rubick AI Gateway")
    parser.add_argument("--host", default=settings.gateway_host, help="Bind host")
    parser.add_argument("--port", type=int, default=settings.gateway_port, help="Bind port")
    parser.add_argument(
        "--backend",
        default=settings.llama_server_url,
        help="llama-server URL (default: http://127.0.0.1:8999)",
    )
    parser.add_argument(
        "--auto-guardian",
        action="store_true",
        default=settings.auto_guardian,
        help="Enable automatic safety pre-flight screening on chat requests",
    )
    parser.add_argument(
        "--intent-parser",
        action=argparse.BooleanOptionalAction,
        default=settings.intent_parser,
        help="Enable/disable automatic intent parsing and head hijacking (default: enabled)",
    )
    parser.add_argument(
        "--explain",
        action=argparse.BooleanOptionalAction,
        default=settings.explain,
        help="Enable/disable natural language chat explanations for decisions (default: enabled)",
    )

    args = parser.parse_args()
    settings.llama_server_url = args.backend
    settings.auto_guardian = args.auto_guardian
    settings.intent_parser = args.intent_parser
    settings.explain = args.explain

    print(f"🪄 Starting Rubick Gateway on {args.host}:{args.port}")
    print(f"🔗 Target llama-server backend: {settings.llama_server_url}")
    print(f"🛡️  Auto-Guardian pre-flight: {'ENABLED' if settings.auto_guardian else 'DISABLED'}")
    print(f"🧠 Intent Parser auto-routing: {'ENABLED' if settings.intent_parser else 'DISABLED'}")
    print(f"💬 Decision explanations: {'ENABLED' if settings.explain else 'DISABLED'}")

    uvicorn.run("rubick.gateway:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
