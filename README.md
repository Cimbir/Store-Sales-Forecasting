# ML Final Project: Store Sales Forecasting


https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting

კონკურში გვევალება ამოვხსნათ Time Series ტიპის ამოცანა. Dataset-ის სახით გადმოგვეცემა თითოეული (Store, Dept) წყვილისათვის Weekly_Sales მიმდევრობა 2010 წლიდან 2011 წლამდე. Weekly_Sales ხასიათდება ყოველწლიური სეზონურობით. ამოცანის მიზანია, რომ გამოვიცნოთ თუ როგორი იქნება Weekly_Sales პარამეტრი შემდეგი 2 წლის განმავლობაში. თითოეულ სტრიქონს dataset-ში დართული აქვს IsHoliday პარამეტრი და ასევე შეგვიძლია გამოვიყენოთ სხვა დამატებითი features `features.csv` და `stores.csv` ფაილებიდან.

კონკურსი ფასდება WMAE მეტრიკით, რაც მსგავსია MAE მეტრიკის, ოღონდ ზოგიერთ მონაცემს, ამ შემთხვევაში `IsHoliday==true` ტიპის მონაცემებს მინიჭებული აქვს წონა 5, დანარჩენებს კი წონა 1.

# Training

ამოცანის ამოსახსნელად გამოვცადეთ რამდენიმე სხვადასხვა მოდელი. თითოეულს გასაწვრთნელად სჭირდება განსხვავებული ტიპის მიდგომა როგორც Feature Selection, Feature Engineering დროს, ასევე თვითონ training პროცესში. შესაბამისად თითოეულ მოდელს განვიხილავთ ცალკე.

ზოგი მოდელის ექსპერიმენტები დალოგილია MLFlow-ზე Dagshub-ის საშუალებით, ზოგი კი Wandb-ზე.

## XGBoost

https://dagshub.com/Cimbir/Store-Sales-Forecasting.mlflow/#/experiments/4

