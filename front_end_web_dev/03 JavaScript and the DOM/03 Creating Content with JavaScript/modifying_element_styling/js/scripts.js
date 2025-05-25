const restyleOtter = async() => {

    const titleDiv = document.querySelector('body').firstElementChild;
    titleDiv.classList.add('title');

    const cardContainer = titleDiv.nextElementSibling;
    cardContainer.id = 'cardContainer'

    const profCard = cardContainer.firstElementChild;
    profCard.style = '';
    profCard.classList.remove('oldProfileCard');
    profCard.classList.add('profileCard');

    const picFrame = profCard.firstElementChild;
    picFrame.classList.add('picFrame');

    const userInfo = picFrame.nextElementSibling;
    userInfo.classList.add('userInfo');

    const statusButton = userInfo.lastElementChild.firstElementChild;
    statusButton.classList.toggle('active');
    statusButton.textContent = `Online now!`;
}

restyleOtter();

