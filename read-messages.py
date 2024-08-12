print("Importando bibliotecas")
from collections import Counter
import can
from gids_dis import Discriminator
import gids_dis as gids
WINDOW_SIZE = 64
THRESHOLD_VALID_FRAME = 0.5

def firstFilter(window_frame):
    # return True
    print("  1.Checking first filter(valids ids)...")
    for id in window_frame:
        if id not in gids.VALID_IDS:
            print("  Invalid id detected: ", id)
            print("--------------------------WE ARE UNDER ATTACK--------------------------")
            return False
    return True

def main():
    #model part
    print("Loading model...")
    model, _, _ = gids.load_model("modelo_cnn64_91.pth", WINDOW_SIZE)

    #can part
    can_interface = 'socketcan'
    channel = 'can0'

    print("Initializing CAN bus...")
    bus = can.interface.Bus(channel=channel, interface=can_interface)

    print(f"Listening for CAN messages on {channel}")

    window_frame = []
    # cnt_messages = 0

    try:
        ignore_frame = False
        while True:
            message = bus.recv()
            # cnt_messages = cnt_messages + 1
            # print(cnt_messages)
            #ignore_frame = not ignore_frame
            #if ignore_frame:
            #   continue
            id = hex(message.arbitration_id).replace("x","").zfill(3).upper()
            window_frame.append(id)
            if len(window_frame) == WINDOW_SIZE:
                print('\n', window_frame)
                valid = firstFilter(window_frame)
                if valid:
                    print("  2.Checking second filter(gids)...")
                    input_data = gids.one_hot_encoded_to_tensor(window_frame)
                    #prediction = model(input_data)
                    prediction = model.forward(input_data)
                    prediction = prediction.detach().item()
                    print('  Prediction of normal message: ', prediction)
                    if prediction < THRESHOLD_VALID_FRAME:
                        print("--------------------------WE ARE UNDER ATTACK--------------------------")
                window_frame = []

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exitting...")

if __name__ == "__main__":
    main()

