package com.example.voiceassistant

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.net.wifi.WifiManager
import android.os.Build
import android.util.Log
import java.net.Inet4Address
import java.net.NetworkInterface

object NetworkUtils {
    private const val TAG = "NetworkUtils"

    /**
     * Get the current WiFi SSID.
     * Requires ACCESS_FINE_LOCATION + ACCESS_WIFI_STATE permissions on Android 10+.
     * Returns null if not connected to WiFi or permissions not granted.
     */
    fun getWifiSsid(context: Context): String? {
        return try {
            val wifiManager = context.applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
            val connectionInfo = wifiManager.connectionInfo
            val ssid = connectionInfo?.ssid

            if (ssid == null || ssid == "<unknown ssid>" || ssid == "0x") {
                null
            } else {
                // SSID is returned with quotes on some devices
                ssid.removePrefix("\"").removeSuffix("\"")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get WiFi SSID: ${e.message}")
            null
        }
    }

    /**
     * Get the device's local IP address on the current network.
     */
    fun getDeviceIp(context: Context): String? {
        return try {
            // Method 1: via WifiManager (most reliable for WiFi)
            val wifiManager = context.applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
            val ipInt = wifiManager.connectionInfo?.ipAddress ?: 0
            if (ipInt != 0) {
                return "${ipInt and 0xFF}.${(ipInt shr 8) and 0xFF}.${(ipInt shr 16) and 0xFF}.${(ipInt shr 24) and 0xFF}"
            }

            // Method 2: via NetworkInterface (fallback)
            NetworkInterface.getNetworkInterfaces()?.toList()
                ?.flatMap { it.inetAddresses.toList() }
                ?.firstOrNull { !it.isLoopbackAddress && it is Inet4Address }
                ?.hostAddress
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get device IP: ${e.message}")
            null
        }
    }

    /**
     * Check if the device is connected to WiFi.
     */
    fun isWifiConnected(context: Context): Boolean {
        val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        val network = cm.activeNetwork ?: return false
        val capabilities = cm.getNetworkCapabilities(network) ?: return false
        return capabilities.hasTransport(NetworkCapabilities.TRANSPORT_WIFI)
    }
}
