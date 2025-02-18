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

## [2.最长连续序列](https://leetcode.cn/problems/longest-consecutive-sequence/description/)
思路：nums用一个hash表来存放，遍历hash表，如果存在比他小1的值，则考虑比他小的值即可，跳过；如果不存在比他小1的值，则说明当前值可能是最长连续序列的起点。  
代码：
```
class Solution {
    public int longestConsecutive(int[] nums) {
        //nums用一个hash表来存放
        Set<Integer> set = new HashSet<>();
        for(int x : nums) 
            set.add(x);
        int maxLength = 0;
        //遍历hash表
        for(int x : set){
            if(set.contains(x-1)) //如果存在比他小1的值，则考虑比他小的值即可。
                continue;
            int t = x; //不存在比他小1的值，说明当前值可能是最长连续序列的起点
            while(set.contains(t))
                t++;
            maxLength = Math.max(maxLength, t - x); //更新答案
        }
        return maxLength;
    }
}
```

# 双指针

## [1.移动零](https://leetcode.cn/problems/move-zeroes/description/)
思路：使用双指针，左指针始终指向第一个0， 右指针指向left后的第一个非零元素（其中left到right-1都为0），进行元素交换后，双指针继续向后移动。  
代码：
```
class Solution {
    public void moveZeroes(int[] nums) {
        int n = nums.length;
        int left = 0; //始终指向第一个0
        while(left < n && nums[left] != 0)
            left++;
        int right = left; //指向left后的非零元素
        while(right < n && nums[right] == 0)
            right++;
        while(right < n){
            int temp = nums[left];
            nums[left] = nums[right];
            nums[right] = temp;
            while(right < n && nums[right] == 0)
                right++;
            while(left < n && nums[left] != 0)
                left++;
        }
    }
}
```
