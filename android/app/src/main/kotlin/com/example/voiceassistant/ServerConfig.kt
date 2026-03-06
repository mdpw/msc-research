package com.example.voiceassistant

import android.content.Context

object ServerConfig {
    private const val PREFS_NAME = "server_config"
    private const val KEY_SERVER_IP = "server_ip"
    private const val KEY_SERVER_PORT = "server_port"

    var serverIp: String = "192.168.1.100"
        private set
    var serverPort: Int = 8000
        private set

    val baseUrl: String get() = "http://$serverIp:$serverPort"

    fun wsUrl(roomNumber: String): String = "ws://$serverIp:$serverPort/ws/guest/$roomNumber"

    fun load(context: Context) {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        serverIp = prefs.getString(KEY_SERVER_IP, "192.168.1.100") ?: "192.168.1.100"
        serverPort = prefs.getInt(KEY_SERVER_PORT, 8000)
    }

    fun saveServer(context: Context, ip: String, port: Int) {
        serverIp = ip
        serverPort = port
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_SERVER_IP, ip)
            .putInt(KEY_SERVER_PORT, port)
            .apply()
    }
}