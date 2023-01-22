import { Notion } from "@neurosity/notion";
import dotenv from 'dotenv';
import * as fs from 'fs';

import { timer } from "rxjs";
import { takeUntil } from "rxjs/operators/index.js";

dotenv.config()

const deviceId = process.env.DEVICE_ID || "";
const email = process.env.EMAIL || "";
const password = process.env.PASSWORD || "";

const verifyEnvs = (email, password, deviceId) => {
  const invalidEnv = (env) => {
    return env === "" || env === 0;
  };
  if (
    invalidEnv(email) ||
    invalidEnv(password) ||
    invalidEnv(deviceId)
  ) {
    console.error(
      "Please verify deviceId, email and password are in .env file, quitting..."
    );
    process.exit(0);
  }
};

const main = async (timeMinutes) => {

  await notion
  .login({
    email,
    password
  }).then(() =>{
    // check that the device is online

  })
  .catch((error) => {
    console.log(error);
    throw new Error(error);
  });
  console.log("Logged in");

  /**
   * Start collecting metrics and writing to files
   */
  // write JSON string to a file
  let fileTimestamp =  Math.floor(Date.now() / 1000);
  
  // e.g CP3, C3, F5, PO3, PO4, F6, C4, CP4
  let channelNames = (await notion.getInfo()).channelNames;

  let timeMillseconds = timeMinutes * 60 * 1000;
  console.log(timeMillseconds)

  /**
   * Get signal quality readings
   */
  let signalQualitySeries = [];
  notion.signalQuality().pipe(
    takeUntil(
      timer(timeMillseconds) // in milliseconds
    )
  ).subscribe(signalQuality => {

    let signalQualityEntry = {
      'unixTimestamp': Math.floor(Date.now() / 1000),
    }
  
    // loop to get a single entry containing power from all channels
    let index = 0;
    for (index; index < channelNames.length; index++) {
      let ch_name = channelNames[index];
      signalQualityEntry[ch_name + "_value"] = signalQuality[index].standardDeviation;
      signalQualityEntry[ch_name + "_status"] = signalQuality[index].status;
    }

    signalQualitySeries.push(signalQualityEntry);
  }).add(() => {
    writeDataToStore("signalQuality", signalQualitySeries, fileTimestamp);
  });;

  /**
   * Record raw brainwaves
   */
  let rawBrainwavesSeries = [];
  notion.brainwaves("raw").pipe(
    takeUntil(
      timer(timeMillseconds) // in milliseconds
    )
  ).subscribe(brainwaves => {
    // get the number of samples in each entry
    let samples = brainwaves.data[0].length

    let index = 0;
    for (index; index < samples; index++) {
      let brainwaveEntry = {};
      brainwaveEntry['index'] = index;
      brainwaveEntry['unixTimestamp'] = brainwaves.info.startTime;
    
      let ch_index = 0;
      for (ch_index; ch_index < channelNames.length; ch_index++) {
        let ch_name = channelNames[ch_index];

        brainwaveEntry[ch_name] = brainwaves.data[ch_index][index];
      }

      // console.log(brainwaveEntry)
      rawBrainwavesSeries.push(brainwaveEntry);
    }
  }).add(() => {
    writeDataToStore("rawBrainwaves", rawBrainwavesSeries, fileTimestamp);
  });

  /**
   * Subscribe to focus metrics
   */
  let focusPredictionSeries = [];
  notion.focus().pipe(
    takeUntil(
      timer(timeMillseconds) // in milliseconds
    )
  ).subscribe(focus => {
    focusPredictionSeries.push(focus);
  }).add(() => {
    writeDataToStore("focus", focusPredictionSeries, fileTimestamp);   
  });


  /**
   * Subscribe to focus metrics
   */
   let calmPredictionSeries = [];
   notion.calm().pipe(
    takeUntil(
      timer(timeMillseconds) // in milliseconds
    )
  ).subscribe(calm => {
     calmPredictionSeries.push(calm);
   }).add(() => {
    writeDataToStore("calm", calmPredictionSeries, fileTimestamp);
  });

   /**
    * Get power by band series
    */
  let powerByBandSeries = [];
  notion.brainwaves("powerByBand").pipe(
    takeUntil(
      timer(timeMillseconds) // in milliseconds
    )
  ).subscribe((brainwaves) => {
    let bandPowerObject = {
      'unixTimestamp': Math.floor(Date.now() / 1000)
    };

    // loop to get a single entry containing power from all channels
    let index = 0;
    for (index; index < channelNames.length; index++) {
      let ch_name = channelNames[index];

      bandPowerObject[ch_name + "_alpha"] = brainwaves.data.alpha[index];
      bandPowerObject[ch_name + "_beta"] = brainwaves.data.beta[index];
      bandPowerObject[ch_name + "_delta"] = brainwaves.data.delta[index];
      bandPowerObject[ch_name + "_gamma"] = brainwaves.data.gamma[index];
      bandPowerObject[ch_name + "_theta"] = brainwaves.data.theta[index];
    }

    powerByBandSeries.push(bandPowerObject);
  }).add(() => {
    writeDataToStore("powerByBand", powerByBandSeries, fileTimestamp);    
  });
};

function writeDataToStore(metric_label, data, recordingStartTimestamp) {
  let outputData = JSON.stringify(data);
    
  let fileTitle = `data/${metric_label}_${recordingStartTimestamp}.json`;
  var createStream = fs.createWriteStream(fileTitle);

  createStream.on('open', () => {
    createStream.write(outputData);
    createStream.end();
  });

  console.log(`Writing data for ${metric_label} complete`);
}



/**
 * Run script
 */
console.log(`${email} attempting to authenticate to ${deviceId}`);
verifyEnvs(email, password, deviceId);

const notion = new Notion({
  deviceId
});

main(10);