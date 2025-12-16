//  cutebot_receiver.ts

// IMPORTANT: Add the cuteBot extension first
// Extensions → search "cuteBot"

radio.setGroup(1)
basic.showIcon(IconNames.Happy)

radio.onReceivedString(function (receivedString) {

    if (receivedString == "GO") {
        cuteBot.motors(80, 80)
        basic.pause(1000)
        cuteBot.motors(0, 0)
    }

    if (receivedString == "WIGGLE") {
        cuteBot.motors(-80, 80)
        basic.pause(300)
        cuteBot.motors(80, -80)
        basic.pause(300)
        cuteBot.motors(0, 0)
    }
})
