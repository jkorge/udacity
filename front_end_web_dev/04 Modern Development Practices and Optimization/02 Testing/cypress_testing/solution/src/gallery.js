import { clearMain } from "./helperFunctions.js";
import city from "../images/city.png";
import city2 from "../images/city2.png";
import greatPlainsLandscape from "../images/greatPlainsLandscape.png";
import lilipads from "../images/lilipads.png";
import ocean from "../images/ocean.png";
import portrait1 from "../images/portrait1.png";
import portrate1 from "../images/portrate1.png";
import snowCoveredMountain from "../images/snowCoveredMountain.png";
import street from "../images/street.png";

const images = [
  city,
  city2,
  greatPlainsLandscape,
  lilipads,
  ocean,
  portrait1,
  portrate1,
  snowCoveredMountain,
  street,
];
export const renderGalleryPage = () => {
  clearMain();
  const div = document.createElement("div");
  div.className = "container";
  div.dataset.cy = "container";

  images.map((url, idx) => {
    const image = document.createElement("img");
    image.src = url;
    //Adds the data attribute to our images
    image.dataset.cy = `image${idx}`;
    image.className = "gallery";
    div.appendChild(image);
  });
  document.querySelector("main").appendChild(div);
};
