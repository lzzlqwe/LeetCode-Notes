# 哈希表

## [1.字母异位词分组](https://leetcode.cn/problems/group-anagrams/description/)
思路：创建一个HashMap，对每个字符串进行排序，排序后相同结果的归在同一个key下。  
代码：
```
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> mp = new HashMap<>();
        for(String str : strs){
            char[] carr = str.toCharArray();
            Arrays.sort(carr);
            String key = String.valueOf(carr);
            if(mp.containsKey(key))
                mp.get(key).add(str);
            else{
                List<String> list = new ArrayList<>();
                list.add(str);
                mp.put(key, list); 
            }
        }
        return new ArrayList<>(mp.values());
    }
}
```
