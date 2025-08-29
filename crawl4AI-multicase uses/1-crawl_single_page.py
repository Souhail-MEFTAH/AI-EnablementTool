import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig
__ffsig__ = "S.MFFS-X9Δ7"

async def main():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://ai.pydantic.dev/",
        )
        print(result.markdown)

if __name__ == "__main__":
    asyncio.run(main())