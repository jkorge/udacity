const rebuildOtter = async() => {
    //add class to title element
    const titleDiv = document.querySelector('body').firstElementChild;
    titleDiv.classList.add('title');

    // add id to card section
    const cardSection = document.querySelector('section')
    cardSection.id = 'cardContainer'

    // create docucment fragment
    const documentFragment = document.createDocumentFragment()

    //create profile card
    const profileCardDiv = document.createElement('div')
    profileCardDiv.classList.add('profileCard')

    //create div for profile pic
    const pictureDiv = document.createElement('div')
    pictureDiv.classList.add('picFrame')
    // create picture for profile pic div
    const picture = document.createElement('img')
    picture.src='./images/otter_profile.webp'
    picture.alt='profilePic'

    // append profile picture to picture div and then to profile card
    pictureDiv.append(picture)
    profileCardDiv.append(pictureDiv)

    // create div for profile text
    const profileDiv = document.createElement('div')
    profileDiv.classList.add('userInfo')

    // create userInfo div
    const userInfoDiv = document.createElement('div')
    const userName = document.createElement('h2')
    userName.textContent = 'Whiskers McOtter'
    const userBio = document.createElement('p')
    userBio.textContent = `Hi! My name is Whiskers McOtter and I'm from Seattle, Washington. Some of my favorite things are Frappuccinos and fish.`

    //append text into user info divs and profile div
    userInfoDiv.append(userName, userBio)
    profileDiv.append(userInfoDiv)

    // create div for button
    const buttonDiv = document.createElement('div')
   const onlineButton = document.createElement('button')
   onlineButton.classList.add('active')
   onlineButton.textContent = 'Online now!'

   // append button and then to profile div
   buttonDiv.append(onlineButton)
   profileDiv.append(buttonDiv)

   // append to profilecard
   profileCardDiv.append(profileDiv)

   // append to the document fragment
   documentFragment.append(profileCardDiv)
   
   //append the fragment to the DOM
   cardSection.append(documentFragment)

}

rebuildOtter();

