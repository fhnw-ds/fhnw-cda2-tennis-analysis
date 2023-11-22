

class DataPipeline:

    def __init__(self):
        pass


    def findCombinations(self, data, framesToBeReached):
        """Finds best combination of object_ids that are seen in more than x frames

        """

        dataBelowThreshold = data[~data['object_id'].isin(self.findValidObject_ids(data, framesToBeReached))]
        value_counts = dataBelowThreshold['object_id'].value_counts().to_frame().reset_index()
        pairs = []
        for i in range(len(value_counts)):
            for j in range(i + 1, len(value_counts)):
                sum_of_pair = value_counts.iloc[i][1] + value_counts.iloc[j][1]
                if (sum_of_pair >= framesToBeReached) and (sum_of_pair <= max(data['frame'])):
                    pairs.append((value_counts.iloc[i][0], value_counts.iloc[j][0], sum_of_pair))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs



    @staticmethod
    def findValidObject_ids(data, framesToBeReached):
        """Finds all object_ids that are seen in more than x frames

        Args:
            df (pd.DataFrame): Dataframe
            framesToBeReached (int): Number of frames to be seen

        Returns:
            list: List of object_ids
        """
        value_counts = data['object_id'].value_counts()
        listOfObject_ids = value_counts[value_counts > framesToBeReached].index.tolist()
        return listOfObject_ids

    @staticmethod
    def overwriteObject_ids(data, playerIdOne, playerIdTwo):
        data.loc[data['object_id'] == playerIdTwo, 'object_id'] = playerIdOne
        return data


    def mergePersons(self, data, thresholdPerson = 0.8, thresholdIds = 0.1):
        """Merges many object_ids into x persons

        Args:
            df (pd.DataFrame): Dataframe
            thresholdPerson (float, optional): Threshold describes the percentage of frames in witch the Persons are allowed not to be seen. Defaults to 0.8.
            thresholdIds (float, optional): Threshold describes the percentage of frames in witch one ID at least must be seen. Defaults to 0.1.

        Returns:
            pd.DataFrame: Dataframe with x persons

        Examples:
            from helperClasses.DataPipeline import DataPipeline
            dp = DataPipeline()
            data = dp.mergePersons(data, thresholdPerson=0.8, thresholdIds=0.1)

            It makes scence to controll the results with the following code:
            data.plot.scatter(x='frame', y='object_id')
        """
        data = self.filterAtrifactIds(data, thresholdIds)
        frame_length = max(data['frame'])
        framesToBeReachedPerson = frame_length * thresholdPerson
        validObject_ids = self.findValidObject_ids(data, framesToBeReachedPerson)
        validPairs = self.findCombinations(data, framesToBeReachedPerson)
        for pair in validPairs:
            data = self.overwriteObject_ids(data, pair[0], pair[1])
            validObject_ids.append(pair[0])

        return data[data['object_id'].isin(validObject_ids)]


    def filterAtrifactIds(self, data, threshold=0.9):
        """Filters out object_ids that are seen in less than x frames

        Args:
            df (pd.DataFrame): Dataframe
            threshold (float, optional): Threshold describes the percentage of frames in witch the object_ids are allowed not to be seen. Defaults to 0.1.

        Returns:
            pd.DataFrame: Dataframe without object_ids that are seen in less than x frames
        """
        frame_length = max(data['frame'])
        framesToBeReached = frame_length * threshold
        validObject_ids = self.findValidObject_ids(data, framesToBeReached)
        return data[data['object_id'].isin(validObject_ids)]

