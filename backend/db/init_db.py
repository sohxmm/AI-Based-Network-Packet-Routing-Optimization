import asyncio
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


async def main():
    from db.database import init_db
    print("🔄 Creating database tables...")
    try:
        await init_db()
        print("✅ Database initialized successfully")
        print("   Tables created: routing_events, network_snapshots, algorithm_metrics, packet_logs")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())