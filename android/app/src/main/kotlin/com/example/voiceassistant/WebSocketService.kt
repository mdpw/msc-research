package com.example.voiceassistant

import android.util.Log
import okhttp3.*
import org.json.JSONObject
import java.util.concurrent.TimeUnit

class WebSocketService(private val roomNumber: String) {

    private val TAG = "WebSocketService"
    private var webSocket: WebSocket? = null
    private var onMessageReceived: ((String) -> Unit)? = null
    private var onStatusUpdate: ((Int, String) -> Unit)? = null

    private val WS_URL = "ws://10.0.2.2:8000/ws/guest/$roomNumber"

    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build()

    fun connect(
        onMessage: (String) -> Unit,
        onStatusChange: ((Int, String) -> Unit)? = null
    ) {
        onMessageReceived = onMessage
        onStatusUpdate = onStatusChange

        val request = Request.Builder()
            .url(WS_URL)
            .build()

        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.d(TAG, "✅ WebSocket connected for Room $roomNumber")
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                Log.d(TAG, "📨 Received raw message: $text")
                try {
                    val json = JSONObject(text)
                    val type = json.optString("type", "")

                    Log.d(TAG, "📨 Message type: $type")

                    when (type) {
                        "status_update" -> {
                            // FIXED: Parse all fields correctly
                            val requestId = json.optInt("request_id", -1)
                            val status = json.optString("status", "")
                            val message = json.optString("message", "")

                            Log.d(TAG, "📨 Status update - ID: $requestId, Status: $status, Message: $message")

                            // Speak the message if available
                            if (message.isNotEmpty()) {
                                onMessageReceived?.invoke(message)
                            }

                            // Update status in UI
                            if (requestId != -1 && status.isNotEmpty()) {
                                onStatusUpdate?.invoke(requestId, status)
                                Log.d(TAG, "✅ Status callback invoked for request $requestId")
                            } else {
                                Log.e(TAG, "❌ Invalid status update data")
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
                webSocket.close(1000, null)
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.e(TAG, "❌ WebSocket error: ${t.message}", t)
            }
        })
    }

    fun disconnect() {
        webSocket?.close(1000, "Goodbye")
        webSocket = null
        Log.d(TAG, "🔌 WebSocket disconnected")
    }
}