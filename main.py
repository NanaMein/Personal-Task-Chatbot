import asyncio
from Agentic_Workflow.async_flow_v1 import personal_task_wrapper

def run():
    # Create a single event loop for the entire session
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        while True:
            print("Starting conversation...")
            input_message = input("Write something: \n\n")
            if input_message.lower() == "exit the loop now":
                break

            # Run async code in the existing loop
            flow = loop.run_until_complete(personal_task_wrapper(input_message=input_message))
            print(flow)
            print("\n" + "*" * 35 + " END " + "*" * 35 + "\n")

    finally:
        # Cleanup when done
        loop.run_until_complete(asyncio.sleep(0))
        loop.close()
        print("Event loop closed properly")



if __name__=='__main__':
    run()