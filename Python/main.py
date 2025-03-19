from wled import *

async def main() -> None:
    """Show example on controlling your WLED device."""
    async with WLED("wled-frenck.local") as led:
        device = await led.update()
        print(device.info.version)

        # Turn strip on, full brightness
        await led.master(on=True, brightness=255)