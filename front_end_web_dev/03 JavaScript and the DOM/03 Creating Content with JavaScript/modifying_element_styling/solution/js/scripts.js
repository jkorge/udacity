const restyleOtter = async() => {
    const titleDiv = document.querySelector('body').firstElementChild;
    console.log(titleDiv);
    titleDiv.classList.add('title');
    const cardContainer = titleDiv.nextElementSibling;
    cardContainer.id = 'cardContainer';
    const profileCard = cardContainer.firstElementChild
    profileCard.classList.remove('oldProfileCard');
    profileCard.classList.add('profileCard');
    profileCard.style.removeProperty('background-color')
    profileCard.style.removeProperty('color')
    const picFrame = profileCard.firstElementChild
    picFrame.classList.add('picFrame')
    const userInfo = picFrame.nextElementSibling
    userInfo.classList.add('userInfo')
    const buttonDiv = userInfo.lastElementChild
    const onlineButton = buttonDiv.firstElementChild
    onlineButton.classList.toggle('active')
    onlineButton.textContent = `Online now!`
}

restyleOtter();

