CREATE DATABASE  IF NOT EXISTS `mes_omi` /*!40100 DEFAULT CHARACTER SET utf8 */;
USE `mes_omi`;
-- MySQL dump 10.13  Distrib 5.7.12, for Win32 (AMD64)
--
-- Host: 127.0.0.1    Database: mes_omi
-- ------------------------------------------------------
-- Server version	5.5.5-10.2.7-MariaDB

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `erp_prod_stb_plan_bt`
--

DROP TABLE IF EXISTS `erp_prod_stb_plan_bt`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `erp_prod_stb_plan_bt` (
  `pri` int(11) NOT NULL DEFAULT 99 COMMENT 'priority: 1~99, but part different',
  `status1` int(11) NOT NULL DEFAULT 1 COMMENT 'complete: 0: wait stb 1:ongoing 9: complete\n\nstatus is mysql keyword, so rename to status1 -- 2020/01/02 lkchena',
  `tool_grp_id` varchar(24) NOT NULL,
  `tool_id` varchar(12) NOT NULL COMMENT 'length extend to 12 for sintering "before/after" omi -- 2020/03/18 lkchena',
  `part_id` varchar(24) NOT NULL,
  `part_raw` varchar(24) DEFAULT NULL,
  `part_name` varchar(24) DEFAULT NULL COMMENT '品名 -- 2019/12/31 lkchena',
  `part_erp_no` varchar(24) DEFAULT NULL COMMENT '料號 -- 2019/12/31 lkchena',
  `mold_id` varchar(16) NOT NULL,
  `powder_type` varchar(16) DEFAULT NULL,
  `powder_erp_no` varchar(24) DEFAULT NULL COMMENT '粉料料號 -- 2019/12/31 lkchena',
  `cnt_plan` int(11) NOT NULL DEFAULT 0,
  `cnt_act` int(11) NOT NULL DEFAULT 0,
  `date_due` varchar(10) NOT NULL COMMENT 'must be key, cause maybe same product get many order -- 2019/12/10 lkchena\nyyyy/mm/dd hh:mm:ss',
  `date_stb` varchar(10) DEFAULT NULL COMMENT 'yyyy/mm/dd hh:mm:ss',
  `date_comp` varchar(10) DEFAULT NULL COMMENT 'yyyy/mm/dd hh:mm:ss',
  `mfg_no` varchar(16) DEFAULT NULL COMMENT '??? why has mfg_no --> Trinity has the mfg_no, why ??? 2019/12/31 lkchena',
  `order_no` varchar(16) DEFAULT NULL,
  `rd_flag` int(11) DEFAULT 0 COMMENT '0: prod 1: rd -- 2020/01/28 lkchena',
  `cust_id` varchar(12) DEFAULT NULL,
  `weight` float DEFAULT 0,
  `unit` int(11) DEFAULT 1 COMMENT 'weight unit: 1: kg 2: ton (tonnage) 2019/12/31 lkchena',
  `owner_id` varchar(16) DEFAULT NULL COMMENT 'order owner, when something wrong, mfg can find sponsor -- 2019/12/31 lkchena',
  `owner_name` varchar(24) DEFAULT NULL,
  `note` varchar(128) DEFAULT NULL,
  `rec_user` varchar(16) DEFAULT NULL,
  `rec_time` varchar(19) DEFAULT NULL,
  `node_id` varchar(16) DEFAULT NULL,
  `node_time` timestamp(6) NULL DEFAULT current_timestamp(6) ON UPDATE current_timestamp(6),
  PRIMARY KEY (`tool_grp_id`,`tool_id`,`part_id`,`mold_id`,`date_due`),
  KEY `idx1` (`tool_id`) COMMENT 'for excel manual import data -- 2020/01/01 lkchena'
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `erp_prod_stb_plan_bt`
--

LOCK TABLES `erp_prod_stb_plan_bt` WRITE;
/*!40000 ALTER TABLE `erp_prod_stb_plan_bt` DISABLE KEYS */;
INSERT INTO `erp_prod_stb_plan_bt` VALUES (1,0,'FORMING-150T','AF15T01','ACQ51F-1',NULL,NULL,NULL,'ACQ51F-1','H065M',NULL,24000,0,'2019/01/01','2019/11/25','','191213-001','123441',0,'000',NULL,1,NULL,NULL,'*','','',NULL,'2020-02-16 01:13:48.355984'),(7,0,'FORMING-150T','AF15T01','ADB58F-1',NULL,NULL,NULL,'ADB58F-1','DR15M',NULL,926,0,'2019/02/05','2019/12/25','',NULL,NULL,0,NULL,NULL,1,NULL,NULL,'*','','',NULL,'2020-02-16 01:13:48.355984'),(6,0,'FORMING-150T','AF15T01','ADB59F',NULL,NULL,NULL,'ADB59F','C15M',NULL,150,0,'2019/02/05','2019/12/25','',NULL,NULL,0,NULL,NULL,1,NULL,NULL,'*','','',NULL,'2020-02-16 01:13:48.355984'),(5,0,'FORMING-150T','AF15T01','ADB60F',NULL,NULL,NULL,'ADB60F','C15M',NULL,778,0,'2019/02/05','2019/12/24','',NULL,NULL,0,NULL,NULL,1,NULL,NULL,'*','','',NULL,'2020-02-16 01:13:48.355984'),(4,0,'FORMING-150T','AF15T01','BBW131F',NULL,NULL,NULL,'BBW131F','C15M',NULL,28847,0,'2019/01/21','2019/12/23','',NULL,NULL,0,NULL,NULL,1,NULL,NULL,'*','','',NULL,'2020-02-16 01:13:48.355984'),(3,0,'FORMING-150T','AF15T01','BBW144F-1',NULL,NULL,NULL,'BBW144F-1','C15M',NULL,14895,0,'2019/01/07','2019/12/03','',NULL,NULL,0,NULL,NULL,1,NULL,NULL,'*','','',NULL,'2020-02-16 01:13:48.355984'),(2,0,'FORMING-150T','AF15T01','BBW47F',NULL,NULL,NULL,'BBW47F','D21M',NULL,1000,0,'2019/12/17','2019/12/03','',NULL,NULL,0,NULL,NULL,1,NULL,NULL,'*','','',NULL,'2020-02-16 01:13:48.355984'),(10,0,'FORMING-20T','AF02T09','SHC-R9-T',NULL,NULL,NULL,'SHC-R9-T','S20M',NULL,12000,0,'2019/12/18','','',NULL,NULL,0,NULL,NULL,1,NULL,NULL,'測試meomo - 2019/12/10 lkchena',NULL,NULL,NULL,'2020-02-16 01:13:48.355984'),(29,0,'FORMING-20T','AF02T09','SHC-RR-T',NULL,NULL,NULL,'SHC-RR-T','S10M',NULL,23500,0,'2019/01/06','2019/11/30','',NULL,NULL,0,NULL,NULL,1,NULL,NULL,'','SYS','2019/12/10 14:34:00',NULL,'2020-02-16 01:13:48.355984'),(25,0,'FORMING-20T','AF02T09','TEST-DATE',NULL,NULL,NULL,'TEST-DATE','D100M',NULL,1200,0,'2019/12/31','','',NULL,NULL,0,NULL,NULL,1,NULL,NULL,'','SYS','2019/12/10 14:43:20',NULL,'2020-02-16 01:13:48.355984'),(20,0,'FORMING-20T','AF02T09','TEST-DATE',NULL,NULL,NULL,'TEST-DATE','D100M',NULL,1200,0,'2020/01/31','','','1','2',0,'3',NULL,1,NULL,NULL,'測試多單-123','SYS','2019/12/11 13:50:11',NULL,'2020-02-16 01:13:48.355984');
/*!40000 ALTER TABLE `erp_prod_stb_plan_bt` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-05-12  8:16:35
