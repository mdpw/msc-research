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
import kotlinx.coroutines.launch
import java.util.*

data class RequestItem(
    val id: Int,
    val requestText: String,
    val intent: String,
    val confidence: Float,
    val status: String,
    val timestamp: String
)

class MainActivity : ComponentActivity() {
    private lateinit var wakeWordService: WakeWordService
    private lateinit var voskService: VoskService
    private lateinit var audioRecorder: AudioRecorder
    private lateinit var nluService: NLUService
    private lateinit var apiService: ApiService
    private lateinit var tts: TextToSpeech
    private lateinit var webSocketService: WebSocketService
    private var ttsReady = false
    private val TAG = "MainActivity"
    private val ROOM_NUMBER = "101"
    private val _requestHistory = mutableStateListOf<RequestItem>()
    private val _wakeWordDetected = mutableStateOf(false)
    private val _wakeWordEnabled = mutableStateOf(true)

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (!isGranted) {
            Toast.makeText(this, "Microphone permission required", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }

        wakeWordService = WakeWordService(this)
        voskService = VoskService(this)
        audioRecorder = AudioRecorder(this)
        nluService = NLUService(this)
        apiService = ApiService()

        lifecycleScope.launch {
            voskService.initialize()
        }

        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts.setLanguage(Locale.US)
                ttsReady = true
            }
        }

        wakeWordService.initialize()
        wakeWordService.startListening(
            onDetected = { wakeWord ->
                Log.d(TAG, "🎯 Wake word detected: $wakeWord")
                runOnUiThread {
                    _wakeWordDetected.value = !_wakeWordDetected.value
                }
            },
            onErrorCallback = { error ->
                Log.e(TAG, "❌ Wake word error: $error")
            }
        )

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
                wakeWordDetected = _wakeWordDetected,
                wakeWordEnabled = _wakeWordEnabled,
                wakeWordService = wakeWordService,
                onSpeakResponse = { message ->
                    if (ttsReady) {
                        tts.speak(message, TextToSpeech.QUEUE_FLUSH, null, null)
                    }
                },
                onAddRequest = { request -> _requestHistory.add(0, request) }
            )
        }
    }

    private fun connectWebSocket() {
        try {
            webSocketService = WebSocketService(ROOM_NUMBER)
            webSocketService.connect(
                onMessage = { message ->
                    runOnUiThread {
                        if (ttsReady) {
                            tts.speak(message, TextToSpeech.QUEUE_FLUSH, null, null)
                        }
                        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
                    }
                },
                onStatusChange = { requestId, status ->
                    runOnUiThread {
                        val index = _requestHistory.indexOfFirst { it.id == requestId }
                        if (index != -1) {
                            val oldRequest = _requestHistory[index]
                            _requestHistory[index] = oldRequest.copy(status = status)

                            if (ttsReady) {
                                val statusMessage = when (status) {
                                    "in_progress" -> "Your request ID $requestId is being processed"
                                    "completed" -> "Your request ID $requestId is completed"
                                    else -> "Your request ID $requestId status is $status"
                                }
                                tts.speak(statusMessage, TextToSpeech.QUEUE_ADD, null, null)
                            }
                        }
                    }
                }
            )
        } catch (e: Exception) {
            Log.e(TAG, "WebSocket error", e)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        wakeWordService.release()
        tts.shutdown()
        voskService.release()
        nluService.close()
        webSocketService.disconnect()
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
    wakeWordDetected: MutableState<Boolean>,
    wakeWordEnabled: MutableState<Boolean>,
    wakeWordService: WakeWordService,
    onSpeakResponse: (String) -> Unit,
    onAddRequest: (RequestItem) -> Unit
) {
    val context = LocalContext.current
    var isRecording by remember { mutableStateOf(false) }
    var isProcessing by remember { mutableStateOf(false) }
    var statusMessage by remember { mutableStateOf("") }
    var lastTranscription by remember { mutableStateOf("") }
    var lastIntent by remember { mutableStateOf("") }
    var lastConfidence by remember { mutableStateOf(0f) }

    // Update status message based on mode
    LaunchedEffect(wakeWordEnabled.value) {
        statusMessage = if (wakeWordEnabled.value) {
            "Listening for 'Hello Hotel'..."
        } else {
            "Tap microphone to start"
        }
    }

    // Wake word detection handler
    LaunchedEffect(wakeWordDetected.value) {
        if (wakeWordEnabled.value && !isRecording && !isProcessing) {
            wakeWordService.stopListening()

            statusMessage = "Wake word detected! Hello Guest, How can I help you?"
            onSpeakResponse("Hello Guest, How can I help you?")

            kotlinx.coroutines.delay(2000)

            processVoiceRequest(
                audioRecorder = audioRecorder,
                voskService = voskService,
                nluService = nluService,
                apiService = apiService,
                roomNumber = roomNumber,
                lifecycleScope = lifecycleScope,
                onRecordingStart = { isRecording = true },
                onRecordingStop = { isRecording = false },
                onProcessingStart = { isProcessing = true },
                onProcessingStop = { isProcessing = false },
                onStatusUpdate = { statusMessage = it },
                onTranscriptionUpdate = { lastTranscription = it },
                onIntentUpdate = { intent, confidence ->
                    lastIntent = intent
                    lastConfidence = confidence
                },
                onSpeakResponse = onSpeakResponse,
                onAddRequest = onAddRequest,
                onComplete = {
                    wakeWordService.startListening(
                        onDetected = { wakeWord ->
                            android.util.Log.d("MainActivity", "🎯 Wake word detected: $wakeWord")
                            wakeWordDetected.value = !wakeWordDetected.value
                        },
                        onErrorCallback = { error ->
                            android.util.Log.e("MainActivity", "❌ Wake word error: $error")
                        }
                    )
                }
            )
        }
    }

    MaterialTheme {
        Surface(
            modifier = Modifier.fillMaxSize(),
            color = MaterialTheme.colorScheme.background
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(16.dp)
            ) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.primaryContainer
                    )
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Text(
                            text = "Hotel Voice Assistant",
                            style = MaterialTheme.typography.headlineMedium,
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = "Room $roomNumber",
                            style = MaterialTheme.typography.titleMedium
                        )
                    }
                }

                Spacer(modifier = Modifier.height(16.dp))

                // Wake Word Toggle
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceVariant
                    )
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column {
                            Text(
                                text = "Wake Word Detection",
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.Bold
                            )
                            Text(
                                text = if (wakeWordEnabled.value) "Say 'Hello Hotel' to activate" else "Use button to record",
                                style = MaterialTheme.typography.bodySmall
                            )
                        }
                        Switch(
                            checked = wakeWordEnabled.value,
                            onCheckedChange = { enabled ->
                                wakeWordEnabled.value = enabled
                                if (enabled) {
                                    wakeWordService.startListening(
                                        onDetected = { wakeWord ->
                                            android.util.Log.d("MainActivity", "🎯 Wake word detected: $wakeWord")
                                            wakeWordDetected.value = !wakeWordDetected.value
                                        },
                                        onErrorCallback = { error ->
                                            android.util.Log.e("MainActivity", "❌ Wake word error: $error")
                                        }
                                    )
                                    statusMessage = "Listening for 'Hello Hotel'..."
                                } else {
                                    wakeWordService.stopListening()
                                    statusMessage = "Tap microphone to start"
                                }
                            }
                        )
                    }
                }

                Spacer(modifier = Modifier.height(16.dp))

                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.secondaryContainer
                    )
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Text(
                            text = when {
                                isRecording -> "🎤 Recording..."
                                isProcessing -> "⚙️ Processing..."
                                else -> "👂 ${if (wakeWordEnabled.value) "Listening" else "Ready"}"
                            },
                            style = MaterialTheme.typography.titleLarge,
                            fontWeight = FontWeight.Bold
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(text = statusMessage)

                        Spacer(modifier = Modifier.height(16.dp))

                        // Microphone Button (only clickable when wake word is OFF)
                        Button(
                            onClick = {
                                if (!wakeWordEnabled.value && !isRecording && !isProcessing) {
                                    lifecycleScope.launch {
                                        statusMessage = "Hello Guest, How can I help you?"
                                        onSpeakResponse("Hello Guest, How can I help you?")

                                        kotlinx.coroutines.delay(2500)

                                        processVoiceRequest(
                                            audioRecorder = audioRecorder,
                                            voskService = voskService,
                                            nluService = nluService,
                                            apiService = apiService,
                                            roomNumber = roomNumber,
                                            lifecycleScope = lifecycleScope,
                                            onRecordingStart = { isRecording = true },
                                            onRecordingStop = { isRecording = false },
                                            onProcessingStart = { isProcessing = true },
                                            onProcessingStop = { isProcessing = false },
                                            onStatusUpdate = { statusMessage = it },
                                            onTranscriptionUpdate = { lastTranscription = it },
                                            onIntentUpdate = { intent, confidence ->
                                                lastIntent = intent
                                                lastConfidence = confidence
                                            },
                                            onSpeakResponse = onSpeakResponse,
                                            onAddRequest = onAddRequest,
                                            onComplete = {
                                                statusMessage = "Tap microphone to start"
                                            }
                                        )
                                    }
                                }
                            },
                            enabled = !wakeWordEnabled.value && !isRecording && !isProcessing,
                            modifier = Modifier.size(100.dp),
                            shape = RoundedCornerShape(50),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = when {
                                    isRecording -> MaterialTheme.colorScheme.error
                                    isProcessing -> MaterialTheme.colorScheme.tertiary
                                    wakeWordEnabled.value -> Color.Gray
                                    else -> MaterialTheme.colorScheme.primary
                                }
                            )
                        ) {
                            Text(
                                text = when {
                                    isRecording -> "⏺️"
                                    isProcessing -> "⚙️"
                                    wakeWordEnabled.value -> "🔒"
                                    else -> "🎤"
                                },
                                fontSize = 40.sp
                            )
                        }

                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = if (wakeWordEnabled.value) "Say 'Hello Hotel' or 'Hi Hotel'" else "Tap to speak",
                            style = MaterialTheme.typography.bodySmall
                        )

                        if (lastTranscription.isNotEmpty()) {
                            Spacer(modifier = Modifier.height(8.dp))
                            Divider()
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(text = "Transcription:")
                            Text(text = "\"$lastTranscription\"", fontWeight = FontWeight.Medium)
                            if (lastIntent.isNotEmpty()) {
                                Spacer(modifier = Modifier.height(4.dp))
                                Text(text = "Intent: $lastIntent (${(lastConfidence * 100).toInt()}%)")
                            }
                        }
                    }
                }

                Spacer(modifier = Modifier.height(16.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        text = "Recent Requests",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )
                    if (requestHistory.isNotEmpty()) {
                        Text(text = "${requestHistory.size} total")
                    }
                }

                Spacer(modifier = Modifier.height(8.dp))

                if (requestHistory.isEmpty()) {
                    Card(
                        modifier = Modifier.fillMaxWidth().weight(1f),
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.surfaceVariant
                        )
                    ) {
                        Box(
                            modifier = Modifier.fillMaxSize(),
                            contentAlignment = Alignment.Center
                        ) {
                            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                Text(text = "📝", fontSize = 48.sp)
                                Spacer(modifier = Modifier.height(8.dp))
                                Text(text = "No requests yet")
                                Text(
                                    text = if (wakeWordEnabled.value) "Say 'Hello Hotel' to start" else "Tap microphone to start",
                                    style = MaterialTheme.typography.bodySmall
                                )
                            }
                        }
                    }
                } else {
                    LazyColumn(
                        modifier = Modifier.fillMaxWidth().weight(1f),
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
}

suspend fun processVoiceRequest(
    audioRecorder: AudioRecorder,
    voskService: VoskService,
    nluService: NLUService,
    apiService: ApiService,
    roomNumber: String,
    lifecycleScope: kotlinx.coroutines.CoroutineScope,
    onRecordingStart: () -> Unit,
    onRecordingStop: () -> Unit,
    onProcessingStart: () -> Unit,
    onProcessingStop: () -> Unit,
    onStatusUpdate: (String) -> Unit,
    onTranscriptionUpdate: (String) -> Unit,
    onIntentUpdate: (String, Float) -> Unit,
    onSpeakResponse: (String) -> Unit,
    onAddRequest: (RequestItem) -> Unit,
    onComplete: () -> Unit
) {
    try {
        onRecordingStart()
        onStatusUpdate("Recording your request...")

        audioRecorder.startRecording()
        kotlinx.coroutines.delay(5000)

        val audioData = audioRecorder.stopRecording()
        onRecordingStop()
        onProcessingStart()
        onStatusUpdate("Processing...")

        lifecycleScope.launch {
            try {
                onStatusUpdate("Transcribing...")
                val transcription = voskService.transcribeAudio(audioData)
                onTranscriptionUpdate(transcription)

                if (transcription.isEmpty()) {
                    onStatusUpdate("No speech detected")
                    onProcessingStop()
                    onComplete()
                    return@launch
                }

                onStatusUpdate("Understanding...")
                val (intent, confidence) = nluService.classifyIntent(transcription)
                onIntentUpdate(intent, confidence)

                onStatusUpdate("Submitting...")
                apiService.submitRequest(
                    roomNumber = roomNumber,
                    requestText = transcription,
                    intent = intent,
                    onSuccess = { response ->
                        onAddRequest(
                            RequestItem(
                                id = response.requestId,
                                requestText = transcription,
                                intent = intent,
                                confidence = confidence,
                                status = "pending",
                                timestamp = getCurrentTime()
                            )
                        )

                        val responseWithId = "Your request ID ${response.requestId} has been received. ${response.message}"
                        onSpeakResponse(responseWithId)
                        onProcessingStop()
                        onComplete()
                    },
                    onError = { error ->
                        android.widget.Toast.makeText(null, error, android.widget.Toast.LENGTH_LONG).show()
                        onProcessingStop()
                        onComplete()
                    }
                )
            } catch (e: Exception) {
                onProcessingStop()
                onComplete()
            }
        }
    } catch (e: Exception) {
        onRecordingStop()
        onProcessingStop()
        onComplete()
    }
}

@Composable
fun RequestCard(request: RequestItem) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(
            modifier = Modifier.fillMaxWidth().padding(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(text = "Request #${request.id}", fontWeight = FontWeight.Bold)
                StatusBadge(status = request.status)
            }
            Spacer(modifier = Modifier.height(8.dp))
            Text(text = request.requestText)
            Spacer(modifier = Modifier.height(4.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(text = "Intent: ${request.intent}")
                Text(text = request.timestamp)
            }
            if (request.confidence > 0) {
                Spacer(modifier = Modifier.height(2.dp))
                Text(text = "Confidence: ${(request.confidence * 100).toInt()}%")
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
        Text(text = text, color = textColor, fontSize = 12.sp, fontWeight = FontWeight.Bold)
    }
}

fun getCurrentTime(): String {
    val calendar = Calendar.getInstance()
    val hour = calendar.get(Calendar.HOUR_OF_DAY)
    val minute = calendar.get(Calendar.MINUTE)
    return String.format("%02d:%02d", hour, minute)
}