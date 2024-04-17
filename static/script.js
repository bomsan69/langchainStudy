async function sendData() {
    const input = document.getElementById("dataInput");
    const summary = document.getElementById("summary");
    const responseElement = document.getElementById("text-container");
  
    responseElement.innerHTML += `
      <div>
          <h4>You</h4>
          <p>
              ${input.value}
          </p>
      </div>
    `;
  
    const response = await fetch("/process", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ data: input.value }),
    });
  
    const result = await response.json();
    if (result.summary) {
      summary.innerText = result.summary;
    }
  
    const chatGPTResponseContainer = document.createElement("div");
    const chatGPTHeading = document.createElement("h4");
    chatGPTHeading.textContent = "ChatGPT";
    const chatGPTMessage = document.createElement("p");
    chatGPTResponseContainer.appendChild(chatGPTHeading);
    chatGPTResponseContainer.appendChild(chatGPTMessage);
    responseElement.appendChild(chatGPTResponseContainer);
  
  
    const words = result.message.split(" ");
    let i = 0;
  
    function typeWord() {
      if (i < words.length) {
        chatGPTMessage.textContent += words[i] + " ";
        i++;
        setTimeout(typeWord, 100); 
      }
    }
  
    typeWord(); 
  }
  