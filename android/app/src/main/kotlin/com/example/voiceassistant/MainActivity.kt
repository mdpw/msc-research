package com.example.voiceassistant

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.voiceassistant.NLUService
import kotlinx.coroutines.launch
import java.util.*

// Data class for request history
data class RequestItem(
    val id: Int,
    val requestText: String,
    val intent: String,
    val confidence: Float,  // Added confidence
    val status: String,
    val timestamp: String
)

// Intent result data class
data class IntentResult(
    val name: String,
    val confidence: Float
)

class MainActivity : ComponentActivity() {

    private lateinit var voskService: VoskService
    private lateinit var audioRecorder: AudioRecorder
    private lateinit var nluService: NLUService
    private lateinit var apiService: ApiService
    private lateinit var tts: TextToSpeech
    private lateinit var webSocketService: WebSocketService
    private var ttsReady = false
    private val TAG = "MainActivity"

    // Fixed room number for device
    private val ROOM_NUMBER = "101"

    // Mutable state for request history that can be updated from WebSocket
    private val _requestHistory = mutableStateListOf<RequestItem>()

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        Log.d(TAG, "🎤 Microphone permission result: $isGranted")
        if (!isGranted) {
            Toast.makeText(this, "Microphone permission required", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        Log.d(TAG, "=".repeat(60))
        Log.d(TAG, "🚀 Starting Hotel Voice Assistant")
        Log.d(TAG, "   Room: $ROOM_NUMBER")
        Log.d(TAG, "=".repeat(60))

        // Request permission
        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }

        // Initialize services
        try {
            Log.d(TAG, "Initializing Vosk service...")
            voskService = VoskService(this)

            Log.d(TAG, "Initializing Audio recorder...")
            audioRecorder = AudioRecorder(this)

            Log.d(TAG, "Initializing NLU service...")
            nluService = NLUService(this)
            // NLU service will run automatic tests on initialization

            Log.d(TAG, "Initializing API service...")
            apiService = ApiService()

            Log.d(TAG, "✅ All services initialized")
        } catch (e: Exception) {
            Log.e(TAG, "❌ Service initialization failed", e)
            Toast.makeText(this, "Failed to initialize services: ${e.message}", Toast.LENGTH_LONG).show()
        }

        // Initialize TTS
        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val result = tts.setLanguage(Locale.US)
                ttsReady = result != TextToSpeech.LANG_MISSING_DATA &&
                        result != TextToSpeech.LANG_NOT_SUPPORTED
                if (ttsReady) {
                    Toast.makeText(this, "✅ TTS Ready", Toast.LENGTH_SHORT).show()
                    Log.d(TAG, "✅ TTS initialized successfully")
                } else {
                    Log.e(TAG, "❌ TTS language not supported")
                }
            } else {
                Log.e(TAG, "❌ TTS initialization failed")
            }
        }

        // Initialize WebSocket
        connectWebSocket()

        setContent {
            VoiceAssistantScreen(
                roomNumber = ROOM_NUMBER,
                voskService = voskService,
                audioRecorder = audioRecorder,
                nluService = nluService,
                apiService = apiService,
                lifecycleScope = lifecycleScope,
                requestHistory = _requestHistory,
                onSpeakResponse = { message -> speakResponse(message) },
                onAddRequest = { request -> _requestHistory.add(0, request) }
            )
        }
    }

    private fun connectWebSocket() {
        try {
            Log.d(TAG, "Connecting to WebSocket...")
            webSocketService = WebSocketService(ROOM_NUMBER)
            webSocketService.connect(
                onMessage = { message ->
                    runOnUiThread {
                        speakResponse(message)
                        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
                        Log.d(TAG, "📱 Received message: $message")
                    }
                },
                onStatusChange = { requestId, status ->
                    runOnUiThread {
                        // Update request status in history
                        val index = _requestHistory.indexOfFirst { it.id == requestId }
                        if (index != -1) {
                            val oldRequest = _requestHistory[index]
                            _requestHistory[index] = oldRequest.copy(status = status)
                            Log.d(TAG, "📱 Request $requestId status updated to: $status")
                        }
                    }
                }
            )
            Log.d(TAG, "✅ WebSocket connected")
        } catch (e: Exception) {
            Log.e(TAG, "❌ WebSocket connection failed", e)
        }
    }

    private fun speakResponse(message: String) {
        if (ttsReady) {
            tts.speak(message, TextToSpeech.QUEUE_FLUSH, null, null)
            Log.d(TAG, "🔊 Speaking: $message")
        } else {
            Log.w(TAG, "⚠️ TTS not ready, cannot speak: $message")
            Toast.makeText(this, "TTS not ready", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()

        Log.d(TAG, "Cleaning up services...")

        try {
            voskService.release()
            Log.d(TAG, "✅ Vosk service released")
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing Vosk", e)
        }

        try {
            nluService.close()  // Changed from release() to close()
            Log.d(TAG, "✅ NLU service released")
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing NLU", e)
        }

        if (::tts.isInitialized) {
            tts.stop()
            tts.shutdown()
            Log.d(TAG, "✅ TTS shutdown")
        }

        if (::webSocketService.isInitialized) {
            webSocketService.disconnect()
            Log.d(TAG, "✅ WebSocket disconnected")
        }

        Log.d(TAG, "👋 MainActivity destroyed")
    }
}

@Composable
fun VoiceAssistantScreen(
    roomNumber: String,
    voskService: VoskService,
    audioRecorder: AudioRecorder,
    nluService: NLUService,
    apiService: ApiService,
    lifecycleScope: kotlinx.coroutines.CoroutineScope,
    requestHistory: List<RequestItem>,
    onSpeakResponse: (String) -> Unit,
    onAddRequest: (RequestItem) -> Unit
) {
    var status by remember { mutableStateOf("Initializing...") }
    var isRecording by remember { mutableStateOf(false) }
    var isInitialized by remember { mutableStateOf(false) }
    var lastTranscription by remember { mutableStateOf("") }
    var lastIntent by remember { mutableStateOf("") }
    var lastConfidence by remember { mutableStateOf(0f) }

    val context = LocalContext.current
    val TAG = "VoiceAssistantScreen"

    // Initialize STT (NLU is already initialized in MainActivity)
    LaunchedEffect(Unit) {
        try {
            Log.d(TAG, "Initializing speech recognition...")
            status = "Initializing speech recognition..."
            voskService.initialize()

            status = "✅ Ready! Press the button to speak"
            isInitialized = true

            Log.d(TAG, "✅ Screen initialized and ready")

            // Optional: Run NLU debug tests
            // Uncomment to see detailed logs for the test phrases
            // nluService.runDebugTests()

        } catch (e: Exception) {
            Log.e(TAG, "❌ Initialization error: ${e.message}", e)
            status = "❌ Error: ${e.message}"
        }
    }

    Surface(
        modifier = Modifier.fillMaxSize(),
        color = MaterialTheme.colorScheme.background
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp)
        ) {
            // Header
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer
                )
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = "🏨 Hotel Voice Assistant",
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "Room $roomNumber",
                        style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.primary
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Recording Button
            Button(
                onClick = {
                    try {
                        if (isRecording) {
                            Log.d(TAG, "🛑 Stopping recording...")
                            isRecording = false
                            status = "Processing audio..."

                            lifecycleScope.launch {
                                try {
                                    // Stop recording and get audio
                                    val audioData = audioRecorder.stopRecording()
                                    Log.d(TAG, "📊 Got ${audioData.size} audio samples")

                                    if (audioData.isEmpty()) {
                                        status = "❌ No audio recorded. Try again."
                                        Log.w(TAG, "⚠️ Empty audio data")
                                    } else {
                                        // Step 1: Transcribe with Vosk
                                        status = "🎤 Transcribing audio..."
                                        val text = voskService.transcribeAudio(audioData)
                                        lastTranscription = text

                                        Log.d(TAG, "📝 Transcription: '$text'")

                                        if (text.isNotEmpty()) {
                                            // Step 2: Classify intent with NLU
                                            status = "🧠 Understanding request..."

                                            // Get intent with confidence
                                            val (intent, confidence) = nluService.classifyIntent(text)
                                            lastIntent = intent
                                            lastConfidence = confidence

                                            Log.d(TAG, "🎯 Intent: $intent (${(confidence * 100).toInt()}% confidence)")

                                            // Check confidence threshold
                                            if (confidence < 0.5f) {
                                                Log.w(TAG, "⚠️ Low confidence: ${(confidence * 100).toInt()}%")
                                                status = "⚠️ Not sure I understood. Intent: $intent (${(confidence * 100).toInt()}%)"
                                                onSpeakResponse("I'm not sure I understood. Did you say $intent?")
                                            } else {
                                                // Step 3: Submit to backend
                                                status = "📤 Sending to backend..."
                                                apiService.submitRequest(
                                                    roomNumber = roomNumber,
                                                    requestText = text,
                                                    intent = intent,
                                                    onSuccess = { response ->
                                                        lifecycleScope.launch {
                                                            // Add to history
                                                            val newRequest = RequestItem(
                                                                id = response.requestId,
                                                                requestText = text,
                                                                intent = intent,
                                                                confidence = confidence,
                                                                status = "pending",
                                                                timestamp = getCurrentTime()
                                                            )
                                                            onAddRequest(newRequest)

                                                            status = "✅ Request sent: $intent"
                                                            onSpeakResponse(response.message)

                                                            Log.d(TAG, "✅ Request submitted successfully")
                                                            Log.d(TAG, "   Request ID: ${response.requestId}")
                                                            Log.d(TAG, "   Message: ${response.message}")
                                                        }
                                                    },
                                                    onError = { error ->
                                                        lifecycleScope.launch {
                                                            status = "❌ Backend error: $error"
                                                            Log.e(TAG, "❌ API error: $error")
                                                        }
                                                    }
                                                )
                                            }

                                        } else {
                                            status = "❌ Could not understand speech. Try again."
                                            Log.w(TAG, "⚠️ Empty transcription from Vosk")
                                        }
                                    }
                                } catch (e: Exception) {
                                    Log.e(TAG, "❌ Processing error", e)
                                    status = "❌ Error: ${e.message}"
                                }
                            }
                        } else {
                            // Start recording
                            Log.d(TAG, "▶️ Starting recording...")
                            if (ContextCompat.checkSelfPermission(
                                    context,
                                    Manifest.permission.RECORD_AUDIO
                                ) == PackageManager.PERMISSION_GRANTED
                            ) {
                                audioRecorder.startRecording()
                                isRecording = true
                                status = "🎤 Listening... Speak now!"
                                lastTranscription = ""
                                lastIntent = ""
                                lastConfidence = 0f
                                Log.d(TAG, "✅ Recording started")
                            } else {
                                status = "❌ Microphone permission required"
                                Log.e(TAG, "❌ Missing microphone permission")
                                Toast.makeText(context, "Please grant microphone permission", Toast.LENGTH_LONG).show()
                            }
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "❌ Button click error", e)
                        status = "❌ Error: ${e.message}"
                        isRecording = false
                    }
                },
                enabled = isInitialized,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(80.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (isRecording)
                        MaterialTheme.colorScheme.error
                    else
                        MaterialTheme.colorScheme.primary
                )
            ) {
                Text(
                    text = if (isRecording) "🛑 Stop Recording" else "🎤 Start Recording",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Status Card
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.secondaryContainer
                )
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = status,
                        style = MaterialTheme.typography.bodyLarge,
                        fontWeight = FontWeight.Medium
                    )

                    // Show last transcription and intent if available
                    if (lastTranscription.isNotEmpty()) {
                        Spacer(modifier = Modifier.height(8.dp))
                        Divider()
                        Spacer(modifier = Modifier.height(8.dp))

                        Text(
                            text = "Transcription:",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSecondaryContainer.copy(alpha = 0.7f)
                        )
                        Text(
                            text = "\"$lastTranscription\"",
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = FontWeight.Medium
                        )

                        if (lastIntent.isNotEmpty()) {
                            Spacer(modifier = Modifier.height(4.dp))
                            Text(
                                text = "Intent: $lastIntent (${(lastConfidence * 100).toInt()}%)",
                                style = MaterialTheme.typography.bodySmall,
                                color = if (lastConfidence > 0.75f)
                                    Color(0xFF155724)
                                else if (lastConfidence > 0.5f)
                                    Color(0xFF856404)
                                else
                                    Color(0xFF721c24)
                            )
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Request History Header
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Recent Requests",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )
                if (requestHistory.isNotEmpty()) {
                    Text(
                        text = "${requestHistory.size} total",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Request History List
            if (requestHistory.isEmpty()) {
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceVariant
                    )
                ) {
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            Text(
                                text = "📝",
                                fontSize = 48.sp
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                text = "No requests yet",
                                style = MaterialTheme.typography.bodyLarge,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                            Text(
                                text = "Start by pressing the microphone button",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f)
                            )
                        }
                    }
                }
            } else {
                LazyColumn(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    items(requestHistory) { request ->
                        RequestCard(request = request)
                    }
                }
            }
        }
    }
}

@Composable
fun RequestCard(request: RequestItem) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Request #${request.id}",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.Bold
                )
                StatusBadge(status = request.status)
            }

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = request.requestText,
                style = MaterialTheme.typography.bodyMedium
            )

            Spacer(modifier = Modifier.height(4.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = "Intent: ${request.intent}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Text(
                    text = request.timestamp,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            // Show confidence
            if (request.confidence > 0) {
                Spacer(modifier = Modifier.height(2.dp))
                Text(
                    text = "Confidence: ${(request.confidence * 100).toInt()}%",
                    style = MaterialTheme.typography.bodySmall,
                    color = if (request.confidence > 0.75f)
                        Color(0xFF155724)
                    else if (request.confidence > 0.5f)
                        Color(0xFF856404)
                    else
                        Color(0xFF721c24)
                )
            }
        }
    }
}

@Composable
fun StatusBadge(status: String) {
    val (backgroundColor, textColor, text) = when (status) {
        "pending" -> Triple(Color(0xFFFFF3CD), Color(0xFF856404), "Pending")
        "in_progress" -> Triple(Color(0xFFD1ECF1), Color(0xFF0C5460), "In Progress")
        "completed" -> Triple(Color(0xFFD4EDDA), Color(0xFF155724), "Completed")
        else -> Triple(Color.Gray, Color.White, status)
    }

    Box(
        modifier = Modifier
            .background(backgroundColor, RoundedCornerShape(12.dp))
            .padding(horizontal = 12.dp, vertical = 4.dp)
    ) {
        Text(
            text = text,
            color = textColor,
            fontSize = 12.sp,
            fontWeight = FontWeight.Bold
        )
    }
}

fun getCurrentTime(): String {
    val calendar = Calendar.getInstance()
    val hour = calendar.get(Calendar.HOUR_OF_DAY)
    val minute = calendar.get(Calendar.MINUTE)
    return String.format("%02d:%02d", hour, minute)
}