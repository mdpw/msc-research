package com.example.voiceassistant

import android.content.Context

data class NetworkProfile(
    val name: String,
    val serverIp: String,
    val serverPort: Int
)

object ServerConfig {
    private const val PREFS_NAME = "server_config"
    private const val KEY_ACTIVE_PROFILE = "active_profile"
    private const val KEY_PROFILE_COUNT = "profile_count"
    private const val KEY_PROFILE_PREFIX = "profile_"

    // Default profiles
    private val DEFAULT_PROFILES = listOf(
        NetworkProfile("Slt Fiber", "192.168.1.158", 8000),
        NetworkProfile("Prolink_AA5B", "192.168.1.100", 8000)
    )

    var profiles: MutableList<NetworkProfile> = DEFAULT_PROFILES.toMutableList()
        private set
    var activeProfileIndex: Int = 0
        private set

    val activeProfile: NetworkProfile get() = profiles[activeProfileIndex]
    val serverIp: String get() = activeProfile.serverIp
    val serverPort: Int get() = activeProfile.serverPort
    val baseUrl: String get() = "http://$serverIp:$serverPort"

    fun wsUrl(roomNumber: String): String = "ws://$serverIp:$serverPort/ws/guest/$roomNumber"

    fun load(context: Context) {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val count = prefs.getInt(KEY_PROFILE_COUNT, 0)

        if (count > 0) {
            profiles.clear()
            for (i in 0 until count) {
                val name = prefs.getString("${KEY_PROFILE_PREFIX}${i}_name", "Profile $i") ?: "Profile $i"
                val ip = prefs.getString("${KEY_PROFILE_PREFIX}${i}_ip", "192.168.1.158") ?: "192.168.1.158"
                val port = prefs.getInt("${KEY_PROFILE_PREFIX}${i}_port", 8000)
                profiles.add(NetworkProfile(name, ip, port))
            }
        }

        activeProfileIndex = prefs.getInt(KEY_ACTIVE_PROFILE, 0).coerceIn(0, profiles.lastIndex)
    }

    fun saveProfiles(context: Context) {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE).edit()
        prefs.putInt(KEY_PROFILE_COUNT, profiles.size)
        profiles.forEachIndexed { i, profile ->
            prefs.putString("${KEY_PROFILE_PREFIX}${i}_name", profile.name)
            prefs.putString("${KEY_PROFILE_PREFIX}${i}_ip", profile.serverIp)
            prefs.putInt("${KEY_PROFILE_PREFIX}${i}_port", profile.serverPort)
        }
        prefs.putInt(KEY_ACTIVE_PROFILE, activeProfileIndex)
        prefs.apply()
    }

    fun switchProfile(context: Context, index: Int) {
        activeProfileIndex = index.coerceIn(0, profiles.lastIndex)
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putInt(KEY_ACTIVE_PROFILE, activeProfileIndex)
            .apply()
    }

    fun updateProfile(context: Context, index: Int, profile: NetworkProfile) {
        if (index in profiles.indices) {
            profiles[index] = profile
            saveProfiles(context)
        }
    }

    fun addProfile(context: Context, profile: NetworkProfile) {
        profiles.add(profile)
        saveProfiles(context)
    }

    fun removeProfile(context: Context, index: Int) {
        if (profiles.size > 1 && index in profiles.indices) {
            profiles.removeAt(index)
            if (activeProfileIndex >= profiles.size) {
                activeProfileIndex = profiles.lastIndex
            }
            saveProfiles(context)
        }
    }
}
