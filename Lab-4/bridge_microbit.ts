//  bridge_microbit.ts

radio.setGroup(1)
serial.redirectToUSB()

basic.showIcon(IconNames.TShirt)

serial.onDataReceived(serial.delimiters(Delimiters.NewLine), function () {
    let cmd = serial.readUntil(serial.delimiters(Delimiters.NewLine))
    cmd = cmd.trim()

    if (cmd == "GO") {
        radio.sendString("GO")
        basic.showArrow(ArrowNames.North)
    }
    else if (cmd == "WIGGLE") {
        radio.sendString("WIGGLE")
        basic.showIcon(IconNames.Snake)
    }
    else {
        serial.writeLine("Unknown cmd: " + cmd)
    }
})
