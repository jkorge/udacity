const handleCardSelection = async (event) => {
    const card = event.target.closest('.product-card');
    alert(card.querySelector('.product-name').textContent)
}

document.querySelector('.product-grid').addEventListener('pointerdown', handleCardSelection);