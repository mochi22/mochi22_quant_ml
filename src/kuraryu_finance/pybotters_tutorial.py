# ccxtよりも拡張性が高いOSS（なので、その分いろいろと面倒もある。一旦はCCXTを使ってみてダメそうならpybottersを試してみる。）

import asyncio

import pybotters


async def main():
    apis = {"bitflyer": ["BITFLYER_API_KEY", "BITFLYER_API_SECRET"]}

    async with pybotters.Client(
        apis=apis, base_url="https://api.bitflyer.com"
    ) as client:
        r = await client.fetch("GET", "/v1/me/getbalance")

        print(r.data)


if __name__ == "__main__":
    asyncio.run(main())