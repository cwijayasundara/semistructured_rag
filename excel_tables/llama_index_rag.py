import warnings
import os
import nest_asyncio
from llama_parse import LlamaParse
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
_ = load_dotenv()

nest_asyncio.apply()

api_key = os.getenv("LLAMA_CLOUD_API_KEY")

parser = LlamaParse(
    api_key=api_key,  # can also be set in your env as LLAMA_CLOUD_API_KEY
    result_type="markdown",
)

documents = parser.load_data("../docs/nvidia_quarterly_revenue_trend_by_market.xlsx")

print(documents)
