const user = {
    firstName: "Whiskers",
    lastName:"McOtter",
    city:"Seattle",
    state:"Washington",
    favorites:["Frappuccinos", "fish"],
    profilePic:"./images/otter_profile.webp"
}

const destructureExercise = async() => {
    const {firstName, lastName, city, state, favorites, profilePic} = user
    const [fave1, fave2] = favorites
    const name = `${firstName} ${lastName}`
    const bio = `Hi! My name is ${name} and I'm from ${city}, ${state}. Some of my favorite things are ${fave1} and ${fave2}.`

    // DO NOT EDIT BELOW THIS LINE!

    const card2 = document.querySelector('.userPop .userInfo')
    const pic2 = document.querySelector('.userPop .picFrame')
    const header = document.createElement('h2')
    const biography = document.createElement('p')
    const pic = document.createElement('img')
    pic.src = `${profilePic}`
    header.textContent = name
    biography.textContent = bio
    pic2.append(pic)
    card2.append(header, biography)

}

destructureExercise()

