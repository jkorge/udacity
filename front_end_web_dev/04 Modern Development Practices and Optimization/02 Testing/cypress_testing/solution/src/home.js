import { clearMain } from "./helperFunctions.js"

export const renderHomePage = () => {
  clearMain()
  console.log("Home")
  document.querySelector("main").innerHTML = `

  <div class="parallax">
    <h1>Professional Phography</h1>
  </div>

  <div class="content">
    <h1>Affordable Prices</h1>
  </div>

  <div class="parallax">
    <h1>Beautify your Life</h1>
  </div>
   <div class="content">
    <button>CONTACT</button>
  </div>

`

  }