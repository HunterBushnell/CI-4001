radio.onReceivedNumber(function (receivedNumber) {
    recent_social_stim = true
    social_time = radio.receivedPacket(RadioPacketProperty.Time)
    social_stim = receivedNumber
    distance = Math.abs(radio.receivedPacket(RadioPacketProperty.SignalStrength) + 28)
    stimulus_level = Math.map(social_stim * sound_stim, 0, 255 * 255, 0, 255)
})
let stimulus_level = 0
let sound_stim = 0
let social_stim = 0
let recent_social_stim = false
let social_time = 0
let distance = 0
let nervous = 50
let afraid = 100
let terrified = 150
let distance_threshold = 100
let distance_records: number[] = []
distance = 0
let prev_distance = 0
let current_time = 0
social_time = 0
let last_social_stim = 0
recent_social_stim = false
social_stim = 0
sound_stim = 0
stimulus_level = 0
let fear_level = 0
loops.everyInterval(500, function () {
    distance_records.push(distance)
    if (distance_records.length >= 2) {
        prev_distance = distance_records.shift()
    }
})
basic.forever(function () {
    sound_stim = input.soundLevel()
    last_social_stim = current_time - social_time
    if (last_social_stim >= 1) {
        stimulus_level = sound_stim
        recent_social_stim = false
    }
    if (stimulus_level >= terrified) {
        fear_level = terrified
        cuteBot.colorLight(cuteBot.RGBLights.ALL, 0xff0000)
    } else if (stimulus_level >= afraid) {
        fear_level = afraid
        cuteBot.colorLight(cuteBot.RGBLights.ALL, 0xff8000)
        music.play(music.builtinPlayableSoundEffect(soundExpression.sad), music.PlaybackMode.UntilDone)
    } else if (stimulus_level >= nervous) {
        fear_level = nervous
        cuteBot.colorLight(cuteBot.RGBLights.ALL, 0xffff00)
        basic.showIcon(IconNames.Sad)
    } else {
        fear_level = input.soundLevel()
        cuteBot.colorLight(cuteBot.RGBLights.ALL, 0x00ff00)
        basic.showIcon(IconNames.Happy)
    }
    radio.sendNumber(fear_level)
})
