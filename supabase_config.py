from supabase import create_client

# Replace these with your Supabase credentials
SUPABASE_URL = "https://viculrdtittnlgikngxg.supabase.co"
SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZpY3VscmR0aXR0bmxnaWtuZ3hnIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTczODEzMzQ1MywiZXhwIjoyMDUzNzA5NDUzfQ.oGZZIK2n5Gs7dxkir603HwOuWmiAdi937cFTvOQKqPA"

supabase = create_client(SUPABASE_URL, SUPABASE_API_KEY)
