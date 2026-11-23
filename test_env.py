
from pathlib import Path
from dotenv import dotenv_values

env_path = Path(__file__).resolve().parent / ".env"

print("ENV file exists:", env_path.exists())
print("ENV file path:", env_path)

values = dotenv_values(env_path)

print("Loaded variable names:", list(values.keys()))

print("GROQ_API_KEY:", bool(values.get("GROQ_API_KEY")))
print("SUPABASE_URL:", bool(values.get("SUPABASE_URL")))
print("SUPABASE_ANON_KEY:", bool(values.get("SUPABASE_ANON_KEY")))
print("SUPABASE_SERVICE_ROLE_KEY:", bool(values.get("SUPABASE_SERVICE_ROLE_KEY")))
