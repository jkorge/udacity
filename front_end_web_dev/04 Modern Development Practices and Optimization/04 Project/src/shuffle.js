const shuffle = (cards) => {
    let current = cards.length;
    const newCardsArray = [...cards];
    while (current != 0) {
        let randomIndex = Math.floor(Math.random() * current);
        current--;

        [newCardsArray[current], newCardsArray[randomIndex]] = [
            newCardsArray[randomIndex],
            newCardsArray[current],
        ];
    }
    return newCardsArray;
};

export { shuffle };
