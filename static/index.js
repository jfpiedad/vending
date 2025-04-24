const DETECTION_DATA_KEYS = ["age", "ageGroup", "weather", "suggestedDrinks"];
const DISPLAY_LABELS = ["Age", "Age Group", "Weather"];

const webSocket = new WebSocket("/ws/vending");
const container = document.querySelector(".right");

webSocket.addEventListener("open", () => {
  console.log("Start");
  webSocket.send("Start");
});

webSocket.addEventListener("close", () => {
  webSocket.send("Close");
});

webSocket.addEventListener("message", (event) => {
  const vending = document.createElement("div");
  const buyOrCancel = document.createElement("div");
  const takeDrink = document.createElement("div");

  vending.className = "vending";
  buyOrCancel.className = "button-area";

  let data = JSON.parse(event.data);

  if (data.messageType == 1) {
    const span = document.createElement("span");
    span.textContent = data.message;

    vending.appendChild(span);
  } else if (data.messageType == 2) {
    takeDrink.style.display = "none";
    buyOrCancel.style.diisplay = "block";

    let detectionData = data.message;

    for (let index = 0; index < 3; index++) {
      const span = document.createElement("span");
      span.textContent = `${DISPLAY_LABELS[index]}: ${
        detectionData[DETECTION_DATA_KEYS[index]]
      }`;
      vending.appendChild(span);
    }

    const span = document.createElement("span");
    span.textContent = "SUGGESTED DRINKS:";
    vending.appendChild(span);

    detectionData[DETECTION_DATA_KEYS[DETECTION_DATA_KEYS.length - 1]].forEach(
      (drink) => {
        const label = document.createElement("label");

        const input = document.createElement("input");
        input.type = "radio";
        input.name = "drink";
        input.className = "drink";
        input.value = drink;

        label.appendChild(input);
        label.append(convertFirstCharOfEachWord(drink));
        vending.appendChild(label);
        vending.appendChild(document.createElement("br")); // Line break after each label
      }
    );

    const btnBuy = document.createElement("input");
    btnBuy.type = "button";
    btnBuy.value = "BUY";
    btnBuy.className = "buy";

    btnBuy.addEventListener("click", () => {
      let drink_options = document.getElementsByClassName("drink");
      let transactionData = {
        age: detectionData["age"],
        ageGroup: convertFirstCharOfEachWord(detectionData["ageGroup"]),
        weather: convertFirstCharOfEachWord(detectionData["weather"]),
      };

      for (let index = 0; index < drink_options.length; index++) {
        if (drink_options[index].checked) {
          transactionData["drinkBought"] = convertFirstCharOfEachWord(
            drink_options[index].value
          );
        }
      }

      webSocket.send(JSON.stringify(transactionData));
    });

    buyOrCancel.appendChild(btnBuy);

    const btnCancel = document.createElement("input");
    btnCancel.type = "button";
    btnCancel.value = "CANCEL";

    btnCancel.addEventListener("click", () => {
      webSocket.send("Cancel");
    });

    buyOrCancel.appendChild(btnCancel);

    vending.appendChild(buyOrCancel);
  } else if (data.messageType == 3) {
    const span = document.createElement("span");
    span.textContent = data.message;

    vending.appendChild(span);
  } else if (data.messageType == 4) {
    const span = document.createElement("span");
    span.textContent = data.message;

    vending.appendChild(span);

    const btnTake = document.createElement("input");
    btnTake.type = "button";
    btnTake.value = "TAKE";

    btnTake.addEventListener("click", () => {
      webSocket.send("Confirm");
    });

    takeDrink.appendChild(btnTake);
    vending.appendChild(takeDrink);
  }

  container.replaceChildren(vending);
});

const convertFirstCharOfEachWord = (text) => {
  let words = text.split(" ");

  for (let index = 0; index < words.length; index++) {
    words[index] = words[index].charAt(0).toUpperCase() + words[index].slice(1);
  }

  return words.join(" ");
};
