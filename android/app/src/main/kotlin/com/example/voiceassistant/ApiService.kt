package com.example.voiceassistant

import android.util.Log
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL
import kotlin.concurrent.thread

data class RequestResponse(
    val success: Boolean,
    val message: String,
    val requestId: Int
)

class ApiService {
    private val TAG = "ApiService"

    // Change this to your server IP
    private val BASE_URL = "http://10.0.2.2:8000"

    fun submitRequest(
        roomNumber: String,
        requestText: String,
        intent: String?,
        onSuccess: (RequestResponse) -> Unit,
        onError: (String) -> Unit
    ) {
        thread {
            try {
                Log.d(TAG, "📤 Sending request to server...")

                val url = URL("$BASE_URL/api/submit-request")
                val connection = url.openConnection() as HttpURLConnection

                connection.requestMethod = "POST"
                connection.setRequestProperty("Content-Type", "application/json")
                connection.doOutput = true
                connection.connectTimeout = 5000
                connection.readTimeout = 5000

                // Create JSON payload
                val json = JSONObject().apply {
                    put("room_number", roomNumber)
                    put("request_text", requestText)
                    if (intent != null) {
                        put("intent", intent)
                    }
                }

                Log.d(TAG, "📝 Payload: $json")

                // Send request
                val writer = OutputStreamWriter(connection.outputStream)
                writer.write(json.toString())
                writer.flush()
                writer.close()

                // Read response
                val responseCode = connection.responseCode
                Log.d(TAG, "📥 Response code: $responseCode")

                if (responseCode == HttpURLConnection.HTTP_OK) {
                    val reader = BufferedReader(InputStreamReader(connection.inputStream))
                    val response = reader.readText()
                    reader.close()

                    Log.d(TAG, "✅ Response: $response")

                    // Parse response
                    val jsonResponse = JSONObject(response)
                    val result = RequestResponse(
                        success = jsonResponse.getBoolean("success"),
                        message = jsonResponse.getString("message"),
                        requestId = jsonResponse.getInt("request_id")
                    )

                    onSuccess(result)
                } else {
                    onError("Server error: $responseCode")
                }

                connection.disconnect()

            } catch (e: Exception) {
                Log.e(TAG, "❌ Error: ${e.message}", e)
                onError(e.message ?: "Unknown error")
            }
        }
    }
}