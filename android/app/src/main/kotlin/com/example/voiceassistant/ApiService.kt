package com.example.voiceassistant

import android.util.Log
import org.json.JSONArray
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

class ApiService(private val baseUrl: String) {
    private val TAG = "ApiService"

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

                val url = URL("$baseUrl/api/submit-request")
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

    fun getRequestHistory(
        roomNumber: String,
        onSuccess: (List<RequestItem>) -> Unit,
        onError: (String) -> Unit
    ) {
        thread {
            try {
                Log.d(TAG, "📤 Fetching history for room $roomNumber...")

                val url = URL("$baseUrl/api/request-history?room_number=$roomNumber")
                val connection = url.openConnection() as HttpURLConnection
                connection.requestMethod = "GET"
                connection.connectTimeout = 5000
                connection.readTimeout = 5000

                val responseCode = connection.responseCode
                if (responseCode == HttpURLConnection.HTTP_OK) {
                    val reader = BufferedReader(InputStreamReader(connection.inputStream))
                    val responseText = reader.readText()
                    reader.close()

                    // FIXED: First parse as JSONObject, then get the "requests" JSONArray
                    val jsonResponse = JSONObject(responseText)
                    val jsonArray = jsonResponse.getJSONArray("requests")
                    
                    val history = mutableListOf<RequestItem>()
                    for (i in 0 until jsonArray.length()) {
                        val obj = jsonArray.getJSONObject(i)
                        history.add(RequestItem(
                            id = obj.getInt("id"),
                            requestText = obj.optString("request_text", ""),
                            intent = obj.optString("intent", "unknown"),
                            confidence = obj.optDouble("confidence", 0.0).toFloat(),
                            department = obj.optString("department", "Pending"),
                            status = obj.optString("status", "pending"),
                            timestamp = obj.optString("created_at", ""),
                            rating = obj.optInt("rating", 0)
                        ))
                    }
                    onSuccess(history)
                } else {
                    onError("Server error: $responseCode")
                }
                connection.disconnect()
            } catch (e: Exception) {
                Log.e(TAG, "History Error: ${e.message}")
                onError(e.message ?: "Unknown error")
            }
        }
    }

    fun cancelRequest(
        requestId: Int,
        roomNumber: String,
        onSuccess: () -> Unit,
        onError: (String) -> Unit
    ) {
        thread {
            try {
                val url = URL("$baseUrl/api/cancel-request")
                val connection = url.openConnection() as HttpURLConnection
                connection.requestMethod = "POST"
                connection.setRequestProperty("Content-Type", "application/json")
                connection.doOutput = true
                connection.connectTimeout = 5000
                connection.readTimeout = 5000

                val json = JSONObject().apply {
                    put("request_id", requestId)
                    put("room_number", roomNumber)
                }

                val writer = OutputStreamWriter(connection.outputStream)
                writer.write(json.toString())
                writer.flush()
                writer.close()

                val responseCode = connection.responseCode
                if (responseCode == HttpURLConnection.HTTP_OK) {
                    val reader = BufferedReader(InputStreamReader(connection.inputStream))
                    val response = reader.readText()
                    reader.close()
                    val jsonResponse = JSONObject(response)
                    if (jsonResponse.getBoolean("success")) {
                        onSuccess()
                    } else {
                        onError(jsonResponse.getString("message"))
                    }
                } else {
                    onError("Server error: $responseCode")
                }
                connection.disconnect()
            } catch (e: Exception) {
                Log.e(TAG, "Cancel Error: ${e.message}")
                onError(e.message ?: "Unknown error")
            }
        }
    }

    fun rateRequest(
        requestId: Int,
        roomNumber: String,
        rating: Int,
        onSuccess: () -> Unit,
        onError: (String) -> Unit
    ) {
        thread {
            try {
                val url = URL("$baseUrl/api/rate-request")
                val connection = url.openConnection() as HttpURLConnection
                connection.requestMethod = "POST"
                connection.setRequestProperty("Content-Type", "application/json")
                connection.doOutput = true
                connection.connectTimeout = 5000
                connection.readTimeout = 5000

                val json = JSONObject().apply {
                    put("request_id", requestId)
                    put("rating", rating)
                    put("room_number", roomNumber)
                }

                val writer = OutputStreamWriter(connection.outputStream)
                writer.write(json.toString())
                writer.flush()
                writer.close()

                val responseCode = connection.responseCode
                if (responseCode == HttpURLConnection.HTTP_OK) {
                    onSuccess()
                } else {
                    onError("Server error: $responseCode")
                }
                connection.disconnect()
            } catch (e: Exception) {
                Log.e(TAG, "Rating Error: ${e.message}")
                onError(e.message ?: "Unknown error")
            }
        }
    }
}