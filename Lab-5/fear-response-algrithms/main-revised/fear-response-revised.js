// --- Globals ------------------------------------------------------------------
radio.onReceivedNumber(function (receivedNumber) {
    // Social fear input (0–255 from other bots)
    social_stim = receivedNumber
    recent_social_stim = true
    lastSocialTime = control.millis()
    // Use RSSI as a distance proxy: smaller = closer
    rssi = radio.receivedPacket(RadioPacketProperty.SignalStrength)
    distance = Math.abs(rssi)
})
let turn_counter = 0
let search_cycles = 0
let fear_level = 0
let stimulus_level = 0
let stim_from_sound = 0
let stim_from_social = 0
let sound_stim = 0
let now = 0
let rssi = 0
let lastSocialTime = 0
let recent_social_stim = false
let social_stim = 0
let distance = 0
// --- Radio + social input ----------------------------------------------------
radio.setGroup(1)
// Fear thresholds (on 0–255 scale)
let nervous = 60
let afraid = 120
let terrified = 180
// Distance and huddle parameters
// current "distance" (from RSSI)
distance = 999
// previous distance snapshot
let prev_distance = 999
// only huddle if within this range
let distance_threshold = 80
// minimum change to consider "closer/farther"
let distance_delta = 3
// ms before we consider social input "stale"
let socialTimeout = 2000
// Periodically snapshot previous distance so we can compare gradients
loops.everyInterval(500, function () {
    prev_distance = distance
})
// --- Main loop ---------------------------------------------------------------
basic.forever(function () {
    now = control.millis()
    // Time-out social stimulus if too old
    if (now - lastSocialTime > socialTimeout) {
        recent_social_stim = false
        social_stim = 0
    }
    // Local sound stimulus (0–255)
    sound_stim = input.soundLevel()
    if (sound_stim > 255) {
        sound_stim = 255
    }
    // Combine inputs: social + sound (simple max)
    stim_from_social = social_stim
    if (stim_from_social > 255) {
        stim_from_social = 255
    }
    stim_from_sound = sound_stim
    stimulus_level = Math.max(stim_from_social, stim_from_sound)
    // --- Fear state machine + behaviors -------------------------------------
    // }
    if (stimulus_level >= terrified) {
        // TERRIFIED
        fear_level = 240
        cuteBot.colorLight(cuteBot.RGBLights.ALL, 0xff0000)
        basic.showIcon(IconNames.Skull)
        if (recent_social_stim && distance < distance_threshold) {
            // --- Huddle mode: reset search counter because we have a target ---
            search_cycles = 0
            if (prev_distance == 999) {
                // No history yet: move forward a bit
                cuteBot.moveTime(cuteBot.Direction.forward, 25, 0.4)
            } else {
                if (distance < prev_distance - distance_delta) {
                    // Closer -> keep going forward, a bit longer
                    cuteBot.moveTime(cuteBot.Direction.forward, 25, 0.4)
                    turn_counter = 0
                } else if (distance > prev_distance + distance_delta) {
                    // Farther -> back up and turn
                    cuteBot.moveTime(cuteBot.Direction.backward, 25, 0.3)
                    cuteBot.turnleft()
                    basic.pause(250)
                    turn_counter = 0
                } else {
                    // No clear gradient: explore mostly forward, occasional turn
                    turn_counter += 1
                    if (turn_counter >= 4) {
                        cuteBot.turnleft()
                        basic.pause(250)
                        turn_counter = 0
                    } else {
                        cuteBot.moveTime(cuteBot.Direction.forward, 20, 0.4)
                    }
                }
            }
        } else {
            // --- TERRIFIED but no nearby fearful bots: search longer before circling ---
            search_cycles += 1
            if (search_cycles <= 6) {
                // First ~6 cycles: move forward to search (6 * 0.4s ≈ 2.4 s)
                cuteBot.moveTime(cuteBot.Direction.forward, 25, 0.4)
            } else {
                // Then do a turn for a bit, then repeat
                cuteBot.turnleft()
                basic.pause(300)
                if (search_cycles >= 10) {
                    search_cycles = 0
                }
            }
        }
    } else if (stimulus_level >= afraid) {
        // AFRAID
        // medium-high fear
        fear_level = 160
        cuteBot.colorLight(cuteBot.RGBLights.ALL, 0xff8000)
        basic.showIcon(IconNames.Sad)
        // Brief sound + small backward retreat
        music.playTone(262, music.beat(BeatFraction.Eighth))
        cuteBot.moveTime(cuteBot.Direction.backward, 20, 0.2)
    } else if (stimulus_level >= nervous) {
        // NERVOUS
        // moderate fear
        fear_level = 80
        cuteBot.colorLight(cuteBot.RGBLights.ALL, 0xffff00)
        basic.showIcon(IconNames.Asleep)
        // Slow wandering forward
        cuteBot.moveTime(cuteBot.Direction.forward, 15, 0.2)
    } else {
        // CALM
        // low fear
        fear_level = 0
        cuteBot.colorLight(cuteBot.RGBLights.ALL, 0x00ff00)
        basic.showIcon(IconNames.Happy)
        cuteBot.stopcar()
    }
    // Broadcast own fear state to others
    radio.sendNumber(fear_level)
})
