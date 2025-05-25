//Webpack Entry Point
import { renderHomePage } from "./home.js";
import { renderNavigation } from "./navigation.js";
//import css
import "../styles/styles.css";
import "../styles/home.css";
import "../styles/heading.css";
import "../styles/gallery.css";

renderNavigation();
renderHomePage();
