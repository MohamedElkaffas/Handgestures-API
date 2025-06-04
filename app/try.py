async function getPredictedLabel(processed_landmarks) {
  try {
    const API_BASE_URL = "http://localhost:8001";
    
    console.log("Calling Hand Gesture API");
    console.log("Landmarks length:", processed_landmarks.length);
    
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        landmarks: processed_landmarks
      })
    });
    
    if (!response.ok) {
      console.error(`API Error: ${response.status} ${response.statusText}`);
      return null;
    }
    
    const data = await response.json();
    console.log("API Response:", data);
    
    // Action mapping
    const actionMapping = {
      "UP": "up",
      "DOWN": "down",
      "LEFT": "left", 
      "RIGHT": "right",
      "STOP": null,
      "WAIT": null,
      "PAUSE": null,
      "OK": null,
      "ACTION": null,
      "MUTE": null,
      "FOUR": null,
      "THREE": null,
      "TWO": null
    };
    
    const maze_action = data.maze_action;
    const frontendAction = actionMapping[maze_action] || null;
    const confidence = data.confidence || 0;
    
    // Confidence thresholds
    const MIN_CONFIDENCE = 0.4;  
    
    // Optional: Different thresholds for different gesture types
    let dynamicThreshold = MIN_CONFIDENCE;
    
    // Stricter for movement (to avoid accidental moves)
    if (["UP", "DOWN", "LEFT", "RIGHT"].includes(maze_action)) {
      dynamicThreshold = 0.45;  // Slightly higher for movement
    }
    // More permissive for stop/wait (easier to trigger)
    else if (["STOP", "WAIT", "PAUSE"].includes(maze_action)) {
      dynamicThreshold = 0.35;  // Lower for stop commands
    }
    
    if (confidence < dynamicThreshold) {
      console.log(`⚠️ Low confidence (${confidence.toFixed(2)} < ${dynamicThreshold}), ignoring gesture`);
      return null;
    }
    
    console.log(`✅ ${data.gesture_name} -> ${maze_action} -> ${frontendAction} (${confidence.toFixed(2)})`);
    return frontendAction;
    
  } catch (error) {
    console.error("❌ API Error:", error);
    return null;
  }
}