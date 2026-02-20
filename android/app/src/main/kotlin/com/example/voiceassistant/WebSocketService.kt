package com.example.voiceassistant

import android.os.Handler
import android.os.Looper
import android.util.Log
import okhttp3.*
import org.json.JSONObject
import java.util.concurrent.TimeUnit

class WebSocketService(private val roomNumber: String, private val wsUrl: String) {

    private val TAG = "WebSocketService"
    private var webSocket: WebSocket? = null
    private var onMessageReceived: ((String) -> Unit)? = null
    private var onStatusUpdate: ((Int, String) -> Unit)? = null

    private var isConnected = false
    private val reconnectHandler = Handler(Looper.getMainLooper())
    private var reconnectAttempts = 0
    private val MAX_RECONNECT_DELAY = 30000L // 30 seconds

    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build()

    fun connect(
        onMessage: (String) -> Unit,
        onStatusChange: ((Int, String) -> Unit)? = null
    ) {
        onMessageReceived = onMessage
        onStatusUpdate = onStatusChange
        internalConnect()
    }

    private fun internalConnect() {
        Log.d(TAG, "🔌 Attempting to connect to WebSocket: $wsUrl")
        val request = Request.Builder()
            .url(wsUrl)
            .build()

        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.d(TAG, "✅ WebSocket connected for Room $roomNumber")
                isConnected = true
                reconnectAttempts = 0
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                Log.d(TAG, "📨 Received raw message: $text")
                try {
                    val json = JSONObject(text)
                    val type = json.optString("type", "")

                    Log.d(TAG, "📨 Message type: $type")

                    when (type) {
                        "status_update" -> {
                            val requestId = json.optInt("request_id", -1)
                            val status = json.optString("status", "")
                            val message = json.optString("message", "")

                            Log.d(TAG, "📨 Status update - ID: $requestId, Status: $status, Message: $message")

                            if (message.isNotEmpty()) {
                                onMessageReceived?.invoke(message)
                            }

                            if (requestId != -1 && status.isNotEmpty()) {
                                onStatusUpdate?.invoke(requestId, status)
                                Log.d(TAG, "✅ Status callback invoked for request $requestId")
                            }
                        }
                        else -> {
                            Log.d(TAG, "⚠️ Unknown message type: $type")
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "❌ Error parsing message: ${e.message}", e)
                }
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                Log.d(TAG, "🔌 WebSocket closing: $reason")
                isConnected = false
                webSocket.close(1000, null)
                scheduleReconnect()
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.e(TAG, "❌ WebSocket error: ${t.message}", t)
                isConnected = false
                scheduleReconnect()
            }
        })
    }

    private fun scheduleReconnect() {
        val delay = Math.min(2000L * (1 shl reconnectAttempts), MAX_RECONNECT_DELAY)
        Log.d(TAG, "🔄 Scheduling reconnect in ${delay}ms (Attempt $reconnectAttempts)")
        
        reconnectHandler.removeCallbacksAndMessages(null)
        reconnectHandler.postDelayed({
            reconnectAttempts++
            internalConnect()
        }, delay)
    }

    fun disconnect() {
        reconnectHandler.removeCallbacksAndMessages(null)
        webSocket?.close(1000, "Goodbye")
        webSocket = null
        isConnected = false
        Log.d(TAG, "🔌 WebSocket disconnected")
    }
}
