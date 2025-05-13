document.addEventListener('DOMContentLoaded', function() {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-message');
    const sendButton = document.getElementById('send-btn');

    function addMessageToChat(message, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = isUser ? 'message user-message' : 'message ai-message';
        
        const paragraph = document.createElement('p');
        paragraph.textContent = message;
        
        messageDiv.appendChild(paragraph);
        chatBox.appendChild(messageDiv);
        
        // Scroll to the bottom of chat
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        // Add user message to chat
        addMessageToChat(message, true);
        
        // Clear input field
        userInput.value = '';
        
        try {
            // Show loading indicator
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'message ai-message loading';
            loadingDiv.innerHTML = '<p>Thinking...</p>';
            chatBox.appendChild(loadingDiv);
            
            // Send message to backend
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: message })
            });
            
            // Remove loading indicator
            chatBox.removeChild(loadingDiv);
            
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            
            const data = await response.json();
            
            // Add AI response to chat
            addMessageToChat(data.response);
            
        } catch (error) {
            console.error('Error:', error);
            addMessageToChat('Sorry, I encountered an error processing your request.');
        }
    }

    // Event listeners
    sendButton.addEventListener('click', sendMessage);
    
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
});
